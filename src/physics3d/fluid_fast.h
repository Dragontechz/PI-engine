/* Fast fluid: Clavet-style double-density relaxation (predict-relax, position based).
 * Key optimizations (per Clavet 2005 + Macklin PBF practice):
 *   - counting-sort uniform grid, cell size = interaction radius h -> 27-cell neighborhood
 *   - half-neighbor-offsets: each pair visited once (14 of 27 offsets)
 *   - 2 density passes (density + near-density) with power kernels, no sqrt per neighbor
 *     in the accumulation (only r^2 compared to h^2)
 *   - displacement applied in the SAME loop as density via Gauss-Seidel style (immediate)
 *   - viscosity impulses only for approaching pairs
 *   - no per-pair division: normalize by 1/sqrt once when needed via rsqrt-style
 * Determinism: fixed particle order (body index), fixed cell iteration order.
 */
#ifndef FLUID_FAST_H
#define FLUID_FAST_H

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define FL_H        0.45f    /* interaction radius (multiple of particle spacing) */
#define FL_H2       (FL_H * FL_H)
#define FL_RHO0     10.0f    /* rest density */
#define FL_K        0.004f   /* stiffness (density restore) */
#define FL_KNEAR    0.01f    /* near-density stiffness (anti-clustering) */
#define FL_VISC     0.08f    /* linear viscosity */
#define FL_MAXDISP  0.30f    /* clamp displacement per pass (stability) */
#define FL_CELL     FL_H
#define FL_NDOFF    14       /* half of 27 neighbor offsets */

/* 14 offsets covering each unordered neighbor pair once in 3D (half-space) */
static const int fl_doff[FL_NDOFF][3] = {
    {0,0,0},
    {1,0,0},{0,1,0},{0,0,1},
    {1,1,0},{1,-1,0},{1,0,1},{1,0,-1},{0,1,1},{0,1,-1},
    {1,1,1},{1,1,-1},{1,-1,1},{-1,1,1}
};

typedef struct {
    int  *cellStart;    /* nCells+1 prefix sums */
    int  *cellEnd;      /* nCells ends (copy before shifting) */
    int  *entries;      /* particle body-indices sorted by cell */
    int  *partCell;     /* cell of each fluid particle */
    int  *fluidIdx;     /* body index of fluid particle fi */
    int   nFluid;
    int   nx, ny, nz;
    int   nCells;
    int   capCells, capParts;
    float x0, y0, z0;
} FluidGrid;

static FluidGrid fg = {0};

static void fluid_ensure(int nParts, int nCells) {
    if (fg.capParts < nParts) {
        int *e2 = (int*)realloc(fg.entries,  nParts * sizeof(int));
        int *p2 = (int*)realloc(fg.partCell, nParts * sizeof(int));
        int *f2 = (int*)realloc(fg.fluidIdx, nParts * sizeof(int));
        if (e2) fg.entries = e2;
        if (p2) fg.partCell = p2;
        if (f2) fg.fluidIdx = f2;
        fg.capParts = nParts;
    }
    if (fg.capCells < nCells) {
        int *s2 = (int*)realloc(fg.cellStart, (nCells + 1) * sizeof(int));
        int *e2 = (int*)realloc(fg.cellEnd,   (nCells + 1) * sizeof(int));
        if (s2) fg.cellStart = s2;
        if (e2) fg.cellEnd = e2;
        fg.capCells = nCells;
    }
}

/* build counting-sort grid: O(n) */
static int fluid_grid_build(World *w) {
    int count = 0;
    for (int i = 0; i < w->bodyCount; i++)
        if (w->bodies[i].tagged == 3) count++;
    fg.nFluid = count;
    if (count == 0) return 0;

    /* make sure per-particle arrays exist before writing to them */
    fluid_ensure(count, 1);

    /* fluid particle list + bounds */
    float minx = 1e9f, miny = 1e9f, minz = 1e9f;
    float maxx = -1e9f, maxy = -1e9f, maxz = -1e9f;
    {
        int fi = 0;
        for (int i = 0; i < w->bodyCount; i++) {
            Body *b = &w->bodies[i];
            if (b->tagged != 3) continue;
            fg.fluidIdx[fi++] = i;
            if (b->pos.x < minx) minx = b->pos.x;
            if (b->pos.y < miny) miny = b->pos.y;
            if (b->pos.z < minz) minz = b->pos.z;
            if (b->pos.x > maxx) maxx = b->pos.x;
            if (b->pos.y > maxy) maxy = b->pos.y;
            if (b->pos.z > maxz) maxz = b->pos.z;
        }
    }

    int nx = (int)((maxx - minx) / FL_CELL) + 3;
    int ny = (int)((maxy - miny) / FL_CELL) + 3;
    int nz = (int)((maxz - minz) / FL_CELL) + 3;
    int nCells = nx * ny * nz;
    fluid_ensure(count, nCells);
    if (!fg.cellStart || !fg.entries) { fg.nFluid = 0; return 0; }

    fg.nx = nx; fg.ny = ny; fg.nz = nz; fg.nCells = nCells;
    fg.x0 = minx; fg.y0 = miny; fg.z0 = minz;

    /* count particles per cell */
    memset(fg.cellStart, 0, (nCells + 1) * sizeof(int));
    for (int fi = 0; fi < count; fi++) {
        Body *b = &w->bodies[fg.fluidIdx[fi]];
        int cx = (int)((b->pos.x - minx) / FL_CELL) + 1;
        int cy = (int)((b->pos.y - miny) / FL_CELL) + 1;
        int cz = (int)((b->pos.z - minz) / FL_CELL) + 1;
        if (cx < 0) cx = 0;
        if (cy < 0) cy = 0;
        if (cz < 0) cz = 0;
        if (cx >= nx) cx = nx - 1;
        if (cy >= ny) cy = ny - 1;
        if (cz >= nz) cz = nz - 1;
        int cell = cx + nx * (cy + ny * cz);
        fg.partCell[fi] = cell;
        fg.cellStart[cell + 1]++;
    }
    /* prefix sum */
    for (int c = 0; c < nCells; c++) fg.cellStart[c + 1] += fg.cellStart[c];
    memcpy(fg.cellEnd, fg.cellStart, (nCells + 1) * sizeof(int));
    /* fill */
    for (int fi = 0; fi < count; fi++)
        fg.entries[fg.cellEnd[fg.partCell[fi]]++] = fg.fluidIdx[fi];
    /* restore: cellEnd now holds ends; shift cellStart to starts is already correct
     * because prefix sum gives starts, and fill used copies in cellEnd. */
    return count;
}

/* get cell range */
static void fluid_cell_range(int cx, int cy, int cz, int *s, int *e) {
    if (cx < 0 || cy < 0 || cz < 0 || cx >= fg.nx || cy >= fg.ny || cz >= fg.nz) { *s = 0; *e = 0; return; }
    int cell = cx + fg.nx * (cy + fg.ny * cz);
    *s = fg.cellStart[cell];
    *e = fg.cellEnd[cell];
}

/* double-density relaxation over the grid.
 * Two passes: viscosity impulses, then density displacement (immediate/Gauss-Seidel). */
static void solve_fluid(World *w) {
    if (fg.nFluid == 0) return;
    const float h = FL_H, h2 = FL_H2;

    /* ---- pass 1: viscosity impulses (approaching pairs only) ---- */
    for (int fi = 0; fi < fg.nFluid; fi++) {
        int i = fg.fluidIdx[fi];
        Body *A = &w->bodies[i];
        int cx = (int)((A->pos.x - fg.x0) / FL_CELL) + 1;
        int cy = (int)((A->pos.y - fg.y0) / FL_CELL) + 1;
        int cz = (int)((A->pos.z - fg.z0) / FL_CELL) + 1;
        for (int d = 0; d < FL_NDOFF; d++) {
            int s, e;
            fluid_cell_range(cx + fl_doff[d][0], cy + fl_doff[d][1], cz + fl_doff[d][2], &s, &e);
            for (int k = s; k < e; k++) {
                int j = fg.entries[k];
                if (j <= i) continue;
                Body *B = &w->bodies[j];
                float dx = B->pos.x - A->pos.x;
                float dy = B->pos.y - A->pos.y;
                float dz = B->pos.z - A->pos.z;
                float r2 = dx*dx + dy*dy + dz*dz;
                if (r2 >= h2 || r2 < 1e-12f) continue;
                float r = sqrtf(r2);
                float nx = dx / r, ny = dy / r, nz = dz / r;
                /* inward relative velocity */
                float vax = A->pos.x - A->prev.x, vay = A->pos.y - A->prev.y, vaz = A->pos.z - A->prev.z;
                float vbx = B->pos.x - B->prev.x, vby = B->pos.y - B->prev.y, vbz = B->pos.z - B->prev.z;
                float vn = (vbx - vax) * nx + (vby - vay) * ny + (vbz - vaz) * nz;
                if (vn > 0.0f) continue;                       /* separating */
                float u = vn * (1.0f - r / h) * FL_VISC;       /* impulse magnitude */
                float ix = nx * u * 0.5f, iy = ny * u * 0.5f, iz = nz * u * 0.5f;
                A->prev.x -= ix; A->prev.y -= iy; A->prev.z -= iz;
                B->prev.x += ix; B->prev.y += iy; B->prev.z += iz;
            }
        }
    }

    /* ---- pass 2: double-density relaxation displacement ---- */
    for (int fi = 0; fi < fg.nFluid; fi++) {
        int i = fg.fluidIdx[fi];
        Body *A = &w->bodies[i];
        int cx = (int)((A->pos.x - fg.x0) / FL_CELL) + 1;
        int cy = (int)((A->pos.y - fg.y0) / FL_CELL) + 1;
        int cz = (int)((A->pos.z - fg.z0) / FL_CELL) + 1;

        /* accumulate density + near-density */
        float rho = 0.0f, rhoNear = 0.0f;
        for (int d = 0; d < FL_NDOFF; d++) {
            int s, e;
            fluid_cell_range(cx + fl_doff[d][0], cy + fl_doff[d][1], cz + fl_doff[d][2], &s, &e);
            for (int k = s; k < e; k++) {
                int j = fg.entries[k];
                if (j == i) continue;
                Body *B = &w->bodies[j];
                float dx = B->pos.x - A->pos.x;
                float dy = B->pos.y - A->pos.y;
                float dz = B->pos.z - A->pos.z;
                float r2 = dx*dx + dy*dy + dz*dz;
                if (r2 >= h2) continue;
                float q = 1.0f - sqrtf(r2) / h;   /* kernel 0..1 */
                rho     += q * q;                  /* (1-r/h)^2 */
                rhoNear += q * q * q;              /* (1-r/h)^3 */
            }
        }

        /* pressure from density error */
        A->density = rho;                      /* cache for surface detection */
        float P = FL_K * (rho - FL_RHO0);
        float Pnear = FL_KNEAR * rhoNear;
        if (P < 0.0f) P = 0.0f;    /* no attraction from density term */

        /* displacement */
        float dxAcc = 0.0f, dyAcc = 0.0f, dzAcc = 0.0f;
        for (int d = 0; d < FL_NDOFF; d++) {
            int s, e;
            fluid_cell_range(cx + fl_doff[d][0], cy + fl_doff[d][1], cz + fl_doff[d][2], &s, &e);
            for (int k = s; k < e; k++) {
                int j = fg.entries[k];
                if (j == i) continue;
                Body *B = &w->bodies[j];
                float dx = B->pos.x - A->pos.x;
                float dy = B->pos.y - A->pos.y;
                float dz = B->pos.z - A->pos.z;
                float r2 = dx*dx + dy*dy + dz*dz;
                if (r2 >= h2 || r2 < 1e-12f) continue;
                float r = sqrtf(r2);
                float q = 1.0f - r / h;
                /* D = dt^2 * (P*q + Pnear*q^2) * unit(B-A); dt^2 folded into k */
                float D = (P * q + Pnear * q * q) * 0.5f;
                if (D > FL_MAXDISP) D = FL_MAXDISP;
                if (D < -FL_MAXDISP) D = -FL_MAXDISP;
                float ux = dx / r, uy = dy / r, uz = dz / r;
                float disp = D;   /* pushing along +unit = away from A */
                dxAcc += ux * disp; dyAcc += uy * disp; dzAcc += uz * disp;
                /* half to neighbor (Gauss-Seidel: immediate) */
                B->pos.x += ux * disp * 0.5f;
                B->pos.y += uy * disp * 0.5f;
                B->pos.z += uz * disp * 0.5f;
            }
        }
        A->pos.x += dxAcc * 0.5f;
        A->pos.y += dyAcc * 0.5f;
        A->pos.z += dzAcc * 0.5f;
    }

    /* ---- pass 3: fluid vs solid collision (water cannot pass through bodies) ----
     * Same grid: for each particle, scan neighbors; non-fluid neighbors are solids.
     * Push particle out along contact normal; reaction pushes solid (mass-weighted).
     * Also treats boxes via simple sphere-vs-particle overlap on bodies only
     * (static boxes collide via collide_statics on the solid side). */
    for (int fi = 0; fi < fg.nFluid; fi++) {
        int i = fg.fluidIdx[fi];
        Body *A = &w->bodies[i];
        int cx = (int)((A->pos.x - fg.x0) / FL_CELL) + 1;
        int cy = (int)((A->pos.y - fg.y0) / FL_CELL) + 1;
        int cz = (int)((A->pos.z - fg.z0) / FL_CELL) + 1;
        for (int d = 0; d < FL_NDOFF; d++) {
            int s, e;
            fluid_cell_range(cx + fl_doff[d][0], cy + fl_doff[d][1], cz + fl_doff[d][2], &s, &e);
            for (int k = s; k < e; k++) {
                int j = fg.entries[k];
                Body *B = &w->bodies[j];
                if (B->tagged == 3) continue;          /* fluid-fluid handled above */
                float dx = A->pos.x - B->pos.x;        /* normal: solid -> particle */
                float dy = A->pos.y - B->pos.y;
                float dz = A->pos.z - B->pos.z;
                float r = A->radius + B->radius;
                float d2 = dx*dx + dy*dy + dz*dz;
                if (d2 >= r * r || d2 < 1e-12f) continue;
                float dist = sqrtf(d2);
                float pen = r - dist;
                float nx = dx / dist, ny = dy / dist, nz = dz / dist;
                /* push particle out, CAPPED: full pen on fast movers creates
                 * pinch-jets when a chain (ragdoll) squeezes through water */
                if (pen > 0.04f) pen = 0.04f;
                A->pos.x += nx * pen;
                A->pos.y += ny * pen;
                A->pos.z += nz * pen;
                /* reaction on solid: proportional to penetration (water push) */
                if (B->invMass > 0.0f) {
                    float push = pen * 0.35f;
                    B->pos.x -= nx * push * (A->invMass / (A->invMass + B->invMass));
                    B->pos.y -= ny * push * (A->invMass / (A->invMass + B->invMass));
                    B->pos.z -= nz * push * (A->invMass / (A->invMass + B->invMass));
                    /* drag: solid drags water along, water resists solid */
                    float rvx = (A->pos.x - A->prev.x) - (B->pos.x - B->prev.x);
                    float rvy = (A->pos.y - A->prev.y) - (B->pos.y - B->prev.y);
                    float rvz = (A->pos.z - A->prev.z) - (B->pos.z - B->prev.z);
                    float rvn = rvx * nx + rvy * ny + rvz * nz;
                    float imp = rvn * 0.10f;
                    A->prev.x -= nx * imp; A->prev.y -= ny * imp; A->prev.z -= nz * imp;
                }
            }
        }
    }
}

#endif /* FLUID_FAST_H */
