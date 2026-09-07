/* Deterministic 3D Verlet physics - core (no rendering). */
#ifndef VERLET3D_H
#define VERLET3D_H

#include "raylib.h"
#include "raymath.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "../../../linalg/linalg.h"

#define MAX_BODIES 2048
#define MAX_CONS 1024
#define MAX_ANG 192
#define MAX_BOXES 16
#define SOLVER_ITERS 16
#define FLUID_NX 14
#define FLUID_NY 10
#define FLUID_NZ 14
#define FIXED_DT (1.0f / 60.0f)   /* half-speed sim */
#define GRAVITY 9.0f
#define MAX_ACCUM 0.25f
#define ARENA 20.0f

typedef struct {
    Vector3 pos, prev, acc;
    float radius;
    float invMass;
    float restitution;
    int tagged;        /* 0 = ball, 1 = ragdoll, 2 = rigid-frame, 3 = fluid */
    /* fluid extras (unused by other body kinds) */
    float density;     /* neighbor density cache, used for surface detection */
} Body;

typedef struct {
    int a, b;
    float restLength;
    float stiffness;
    int thick;         /* draw as fat limb */
    int hidden;
} Constraint;

/* angular joint limit: angle at joint 'b' between a-b and b-c must stay <= maxAngle */
typedef struct {
    int a, b, c;
    float maxAngle;
    float stiffness;
} AngCon;

typedef struct {
    Vector3 center, half;
    float yaw;
    Color col;
} StaticBox;

/* ---- water surface field (per-world, declared before World) ---- */
#define WAV_N 32
#define WAV_CELL (2.0f * ARENA / WAV_N)
typedef struct { float h; float vy; int count; float foam; int wet; float sum; } WaveCell;

typedef struct {
    Body bodies[MAX_BODIES];
    int bodyCount;
    Constraint cons[MAX_CONS];
    int conCount;
    AngCon angs[MAX_ANG];
    int angCount;
    StaticBox boxes[MAX_BOXES];
    int boxCount;
    unsigned long long rngState;
    unsigned long long stepCounter;
    double accumulator;
    WaveCell waveGrid[WAV_N][WAV_N];   /* per-world water surface field */
} World;

/* deterministic PRNG xorshift64* */
static unsigned long long rng_next(World *w) {
    unsigned long long x = w->rngState;
    x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
    w->rngState = x;
    return x * 0x2545F4914F6CDD1DULL;
}

static float rng_range(World *w, float lo, float hi) {
    return lo + (hi - lo) * ((float)(rng_next(w) >> 40) / (float)(1 << 24));
}

static void rng_seed(World *w, unsigned long long seed) {
    w->rngState = seed ? seed : 0x9E3779B97F4A7C15ULL;
    for (int i = 0; i < 8; i++) (void)rng_next(w);
}

static int add_body(World *w, float x, float y, float z, float r, float invMass, float rest, int tagged) {
    if (w->bodyCount >= MAX_BODIES) return -1;
    Body *b = &w->bodies[w->bodyCount];
    b->pos = (Vector3){x, y, z};
    b->prev = b->pos;
    b->acc = (Vector3){0, 0, 0};
    b->radius = r;
    b->invMass = invMass;
    b->restitution = rest;
    b->tagged = tagged;
    return w->bodyCount++;
}

static void add_con(World *w, int a, int b, float stiff, int thick, int hidden) {
    if (w->conCount >= MAX_CONS || a < 0 || b < 0) return;
    Constraint *c = &w->cons[w->conCount++];
    c->a = a; c->b = b;
    Vector3 d = Vector3Subtract(w->bodies[b].pos, w->bodies[a].pos);
    c->restLength = Vector3Length(d);
    c->stiffness = stiff;
    c->thick = thick;
    c->hidden = hidden;
}

/* rigid tetrahedron: 4 particles + 6 edges */
static void spawn_tetra(World *w, float cx, float cy, float cz, float size, float yaw) {
    int idx[4];
    Vector3 v[4] = {{1,1,1},{-1,-1,1},{-1,1,-1},{1,-1,-1}};
    float ca = cosf(yaw), sa = sinf(yaw);
    for (int i = 0; i < 4; i++) {
        float x = v[i].x * size, z = v[i].z * size;
        idx[i] = add_body(w, cx + x * ca - z * sa, cy + v[i].y * size, cz + x * sa + z * ca,
                          0.12f, 1.0f, 0.15f, 2);
    }
    for (int i = 0; i < 4; i++)
        for (int j = i + 1; j < 4; j++)
            add_con(w, idx[i], idx[j], 1.0f, 0, 0);
}

/* rigid cube: 8 corners, 12 edges + hidden braces + space diagonals */
static void spawn_cube(World *w, float cx, float cy, float cz, float hx, float hy, float hz, float yaw) {
    int idx[8];
    float ca = cosf(yaw), sa = sinf(yaw);
    for (int i = 0; i < 8; i++) {
        float x = (i & 1 ? hx : -hx);
        float y = (i & 2 ? hy : -hy);
        float z = (i & 4 ? hz : -hz);
        idx[i] = add_body(w, cx + x * ca - z * sa, cy + y, cz + x * sa + z * ca,
                          0.12f, 1.0f, 0.15f, 2);
    }
    for (int i = 0; i < 8; i++)
        for (int j = i + 1; j < 8; j++) {
            int d = i ^ j;
            if (d == 1 || d == 2 || d == 4) add_con(w, idx[i], idx[j], 1.0f, 0, 0);
        }
    add_con(w, idx[0], idx[3], 1.0f, 0, 1);
    add_con(w, idx[1], idx[2], 1.0f, 0, 1);
    add_con(w, idx[4], idx[7], 1.0f, 0, 1);
    add_con(w, idx[5], idx[6], 1.0f, 0, 1);
    add_con(w, idx[0], idx[5], 1.0f, 0, 1);
    add_con(w, idx[1], idx[4], 1.0f, 0, 1);
    add_con(w, idx[0], idx[7], 1.0f, 0, 1);
    add_con(w, idx[1], idx[6], 1.0f, 0, 1);
}

/* angular joint limit: keeps the angle at vertex B from folding past maxAngle */
static void add_ang(World *w, int a, int b, int c, float maxAngle, float stiffness) {
    if (w->angCount >= MAX_ANG || a < 0 || b < 0 || c < 0) return;
    AngCon *ac = &w->angs[w->angCount++];
    ac->a = a; ac->b = b; ac->c = c;
    ac->maxAngle = maxAngle;
    ac->stiffness = stiffness;
}

/* 3D ragdoll (Y-up: head above, feet below).
 * Realistic feel: heavy torso/head, light limbs, joint angle limits,
 * shoulder/pelvis width bracing. */
static void spawn_ragdoll(World *w, float x, float y, float z, float s) {
    float mTorso = 4.0f, mHead = 2.5f, mLimb = 0.8f;
    int head  = add_body(w, x, y,                z, 0.30f*s, 1.0f/mHead, 0.02f, 1);
    int shoul = add_body(w, x, y - 0.55f*s,      z, 0.16f*s, 1.0f/mTorso, 0.02f, 1);
    int hip   = add_body(w, x, y - 1.30f*s,      z, 0.16f*s, 1.0f/mTorso, 0.02f, 1);
    int elbL  = add_body(w, x - 0.45f*s, y - 0.90f*s, z, 0.10f*s, 1.0f/mLimb, 0.02f, 1);
    int elbR  = add_body(w, x + 0.45f*s, y - 0.90f*s, z, 0.10f*s, 1.0f/mLimb, 0.02f, 1);
    int hndL  = add_body(w, x - 0.60f*s, y - 1.45f*s, z, 0.10f*s, 1.0f/mLimb, 0.02f, 1);
    int hndR  = add_body(w, x + 0.60f*s, y - 1.45f*s, z, 0.10f*s, 1.0f/mLimb, 0.02f, 1);
    int kneL  = add_body(w, x - 0.25f*s, y - 2.05f*s, z, 0.10f*s, 1.0f/mLimb, 0.02f, 1);
    int kneR  = add_body(w, x + 0.25f*s, y - 2.05f*s, z, 0.10f*s, 1.0f/mLimb, 0.02f, 1);
    int fotL  = add_body(w, x - 0.30f*s, y - 2.70f*s, z, 0.10f*s, 1.0f/mLimb, 0.02f, 1);
    int fotR  = add_body(w, x + 0.30f*s, y - 2.70f*s, z, 0.10f*s, 1.0f/mLimb, 0.02f, 1);

    add_con(w, head,  shoul, 1.0f, 1, 0);
    add_con(w, shoul, hip,   1.0f, 1, 0);
    add_con(w, shoul, elbL,  1.0f, 1, 0);
    add_con(w, elbL,  hndL,  1.0f, 1, 0);
    add_con(w, shoul, elbR,  1.0f, 1, 0);
    add_con(w, elbR,  hndR,  1.0f, 1, 0);
    add_con(w, hip,   kneL,  1.0f, 1, 0);
    add_con(w, kneL,  fotL,  1.0f, 1, 0);
    add_con(w, hip,   kneR,  1.0f, 1, 0);
    add_con(w, kneR,  fotR,  1.0f, 1, 0);
    add_con(w, head,  hip,   0.35f, 0, 1);   /* stiffer spine brace */

    /* joint angle limits (max angle at middle joint, radians) */
    add_ang(w, head, shoul, elbL, 2.4f, 0.5f);   /* shoulder L */
    add_ang(w, head, shoul, elbR, 2.4f, 0.5f);   /* shoulder R */
    add_ang(w, shoul, elbL, hndL, 2.2f, 0.6f);   /* elbow L */
    add_ang(w, shoul, elbR, hndR, 2.2f, 0.6f);   /* elbow R */
    add_ang(w, shoul, hip, kneL, 2.0f, 0.6f);    /* hip L */
    add_ang(w, shoul, hip, kneR, 2.0f, 0.6f);    /* hip R */
    add_ang(w, hip, kneL, fotL, 1.8f, 0.7f);     /* knee L */
    add_ang(w, hip, kneR, fotR, 1.8f, 0.7f);     /* knee R */
    add_ang(w, shoul, head, hip, 2.0f, 0.6f);    /* neck */
}

static void spawn_water(World *w, float x0, float y0, float z0, int nx, int ny, int nz, float spacing);

static void world_reset(World *w, unsigned long long seed) {
    memset(w, 0, sizeof(*w));
    rng_seed(w, seed);

    w->boxes[w->boxCount++] = (StaticBox){{-6, 0.4f, -2}, {4, 0.4f, 2.5f},  0.35f, (Color){70, 90, 110, 255}};
    w->boxes[w->boxCount++] = (StaticBox){{ 6, 0.4f,  2}, {4, 0.4f, 2.5f}, -0.35f, (Color){70, 90, 110, 255}};
    w->boxes[w->boxCount++] = (StaticBox){{ 0, 2.0f,  0}, {1.5f, 0.4f, 1.5f}, 0.0f, (Color){70, 90, 110, 255}};
    /* pool walls: 4 low walls forming a basin around the water spawn area
     * (StaticBox: center, half-extents, yaw, color) */
    w->boxes[w->boxCount++] = (StaticBox){{ 0, 1.6f, -6.5f}, {9.0f, 1.6f, 0.3f}, 0.0f, (Color){60, 80, 100, 255}};
    w->boxes[w->boxCount++] = (StaticBox){{ 0, 1.6f,  6.5f}, {9.0f, 1.6f, 0.3f}, 0.0f, (Color){60, 80, 100, 255}};
    w->boxes[w->boxCount++] = (StaticBox){{-9.5f, 1.6f, 0}, {0.3f, 1.6f, 6.5f}, 0.0f, (Color){60, 80, 100, 255}};
    w->boxes[w->boxCount++] = (StaticBox){{ 9.5f, 1.6f, 0}, {0.3f, 1.6f, 6.5f}, 0.0f, (Color){60, 80, 100, 255}};

    for (int i = 0; i < 3; i++)
        spawn_tetra(w, rng_range(w, -8, 8), rng_range(w, 15, 25), rng_range(w, -8, 8),
                    rng_range(w, 0.5f, 0.9f), rng_range(w, 0, 6.28f));
    for (int i = 0; i < 3; i++)
        spawn_cube(w, rng_range(w, -8, 8), rng_range(w, 20, 32), rng_range(w, -8, 8),
                   rng_range(w, 0.4f, 0.8f), rng_range(w, 0.4f, 0.8f), rng_range(w, 0.4f, 0.8f),
                   rng_range(w, 0, 6.28f));
    spawn_ragdoll(w, rng_range(w, -5, -2), rng_range(w, 14, 20), rng_range(w, -3, 3), 1.0f);
    spawn_ragdoll(w, rng_range(w,  2,  5), rng_range(w, 20, 28), rng_range(w, -3, 3), 1.1f);

    /* pool: water spawns in a wall-less center basin (dam-break block) */
    spawn_water(w, -7.0f, 1.0f, -3.0f, FLUID_NX, FLUID_NY, FLUID_NZ, 0.30f);

    for (int r = 0; r < 4; r++)
        for (int c = 0; c <= r; c++)
            for (int d = 0; d <= c; d++)
                add_body(w, (c - r * 0.5f) * 0.7f, 2.0f + r * 0.62f, (d - c * 0.5f) * 0.7f,
                         0.3f, 1.0f, 0.2f, 0);

    int prevId = -1;
    for (int i = 0; i < 8; i++) {
        int id = add_body(w, -12.0f, 6.0f - i * 0.6f, -10.0f, 0.12f,
                          i == 0 ? 0.0f : 1.0f, 0.1f, 0);
        if (prevId >= 0) add_con(w, prevId, id, 1.0f, 0, 0);
        prevId = id;
    }
    w->accumulator = 0.0;
}

static void apply_gravity(World *w) {
    for (int i = 0; i < w->bodyCount; i++)
        if (w->bodies[i].invMass != 0.0f) w->bodies[i].acc.y -= GRAVITY; /* Y-up world: gravity pulls down */
}

static void verlet_integrate(World *w) {
    const float dt = FIXED_DT;
    const float damp = 0.998f;
    for (int i = 0; i < w->bodyCount; i++) {
        Body *b = &w->bodies[i];
        if (b->invMass == 0.0f) { b->acc = (Vector3){0, 0, 0}; continue; }
        Vector3 vel = Vector3Subtract(b->pos, b->prev);
        /* ragdoll parts get extra joint/muscle damping so they settle */
        float d = (b->tagged == 1) ? 0.97f : damp;
        b->prev = b->pos;
        b->pos.x += vel.x * d + b->acc.x * dt * dt;
        b->pos.y += vel.y * d + b->acc.y * dt * dt;
        b->pos.z += vel.z * d + b->acc.z * dt * dt;
        b->acc = (Vector3){0, 0, 0};
    }
}

static void solve_constraints(World *w) {
    /* skip fluid in stick constraints (it interacts via solve_fluid) */
    for (int it = 0; it < SOLVER_ITERS; it++) {
        for (int i = 0; i < w->conCount; i++) {
            Constraint *c = &w->cons[i];
            Body *A = &w->bodies[c->a];
            Body *B = &w->bodies[c->b];
            if (A->tagged == 3 || B->tagged == 3) continue;
            float dx = B->pos.x - A->pos.x;
            float dy = B->pos.y - A->pos.y;
            float dz = B->pos.z - A->pos.z;
            float d2 = dx*dx + dy*dy + dz*dz;
            if (d2 < 1e-8f) continue;
            float d = sqrtf(d2);
            float diff = (d - c->restLength) / d * c->stiffness;
            float wA = A->invMass, wB = B->invMass, wSum = wA + wB;
            if (wSum == 0.0f) continue;
            A->pos.x += dx * diff * (wA / wSum);
            A->pos.y += dy * diff * (wA / wSum);
            A->pos.z += dz * diff * (wA / wSum);
            B->pos.x -= dx * diff * (wB / wSum);
            B->pos.y -= dy * diff * (wB / wSum);
            B->pos.z -= dz * diff * (wB / wSum);
        }
    }
}

/* Joint limits as MIN-CHORD constraints:
 * if the angle at joint B exceeds maxAngle, the distance |AC| shrinks below
 * the chord length corresponding to maxAngle. So we just enforce
 * d(A,C) >= chord_max. This is a pure distance constraint -> fully
 * compatible with the stick solver, no fighting, no jitter. */
static void solve_angular(World *w) {
    for (int it = 0; it < SOLVER_ITERS; it++) {
        for (int i = 0; i < w->angCount; i++) {
            AngCon *ac = &w->angs[i];
            Body *A = &w->bodies[ac->a];
            Body *B = &w->bodies[ac->b];
            Body *C = &w->bodies[ac->c];
            float bax = A->pos.x - B->pos.x, bay = A->pos.y - B->pos.y, baz = A->pos.z - B->pos.z;
            float bcx = C->pos.x - B->pos.x, bcy = C->pos.y - B->pos.y, bcz = C->pos.z - B->pos.z;
            float la = sqrtf(bax*bax + bay*bay + baz*baz);
            float lc = sqrtf(bcx*bcx + bcy*bcy + bcz*bcz);
            if (la < 1e-8f || lc < 1e-8f) continue;
            float dot = (bax*bcx + bay*bcy + baz*bcz) / (la * lc);
            if (dot > 1.0f) dot = 1.0f;
            if (dot < -1.0f) dot = -1.0f;
            float ang = acosf(dot);
            if (ang <= ac->maxAngle) continue;

            /* required minimum chord for maxAngle with current segment lengths */
            float chordMin = sqrtf(la*la + lc*lc - 2.0f*la*lc*cosf(ac->maxAngle));
            float dx = C->pos.x - A->pos.x, dy = C->pos.y - A->pos.y, dz = C->pos.z - A->pos.z;
            float d = sqrtf(dx*dx + dy*dy + dz*dz);
            if (d >= chordMin || d < 1e-8f) continue;
            float pen = (chordMin - d) * ac->stiffness;
            float nx = dx / d, ny = dy / d, nz = dz / d;
            float wA = A->invMass, wC = C->invMass, wSum = wA + wC;
            if (wSum == 0.0f) continue;
            A->pos.x -= nx * pen * (wA / wSum);
            A->pos.y -= ny * pen * (wA / wSum);
            A->pos.z -= nz * pen * (wA / wSum);
            C->pos.x += nx * pen * (wC / wSum);
            C->pos.y += ny * pen * (wC / wSum);
            C->pos.z += nz * pen * (wC / wSum);
        }
    }
}

static void collide_static_box(Body *b, StaticBox *bx) {
    float ca = cosf(-bx->yaw), sa = sinf(-bx->yaw);
    float px = b->pos.x - bx->center.x;
    float pz = b->pos.z - bx->center.z;
    float lx = px * ca - pz * sa;
    float lz = px * sa + pz * ca;
    float ly = b->pos.y - bx->center.y;

    float qx = lx, qy = ly, qz = lz;
    if (qx >  bx->half.x) qx =  bx->half.x;
    if (qx < -bx->half.x) qx = -bx->half.x;
    if (qy >  bx->half.y) qy =  bx->half.y;
    if (qy < -bx->half.y) qy = -bx->half.y;
    if (qz >  bx->half.z) qz =  bx->half.z;
    if (qz < -bx->half.z) qz = -bx->half.z;

    float dx = lx - qx, dy = ly - qy, dz = lz - qz;
    float d2 = dx*dx + dy*dy + dz*dz;
    float r = b->radius + 0.05f;
    if (d2 >= r * r) return;

    float d = sqrtf(d2);
    float nx, ny, nz;
    if (d < 1e-6f) {
        float ox = bx->half.x - fabsf(lx);
        float oy = bx->half.y - fabsf(ly);
        float oz = bx->half.z - fabsf(lz);
        if (oy <= ox && oy <= oz) { nx = 0; ny = ly > 0 ? 1 : -1; nz = 0; }
        else if (ox <= oz)        { nx = lx > 0 ? 1 : -1; ny = 0; nz = 0; }
        else                      { nx = 0; ny = 0; nz = lz > 0 ? 1 : -1; }
        d = 0.0f;
    } else {
        nx = dx / d; ny = dy / d; nz = dz / d;
    }
    float pen = r - d;
    float ca2 = cosf(bx->yaw), sa2 = sinf(bx->yaw);
    float wnx = nx * ca2 - nz * sa2;
    float wnz = nx * sa2 + nz * ca2;
    b->pos.x += wnx * pen;
    b->pos.y += ny * pen;
    b->pos.z += wnz * pen;
    b->prev.x += wnx * pen * 0.7f;
    b->prev.y += ny * pen * 0.7f;
    b->prev.z += wnz * pen * 0.7f;
    float vx = b->pos.x - b->prev.x, vy = b->pos.y - b->prev.y, vz = b->pos.z - b->prev.z;
    float vn = vx * wnx + vy * ny + vz * wnz;
    b->prev.x -= (vx - vn * wnx) * 0.15f;
    b->prev.y -= (vy - vn * ny) * 0.15f;
    b->prev.z -= (vz - vn * wnz) * 0.15f;
}

static void collide_statics(World *w) {
    for (int i = 0; i < w->bodyCount; i++) {
        if (w->bodies[i].invMass == 0.0f) continue;
        for (int k = 0; k < w->boxCount; k++)
            collide_static_box(&w->bodies[i], &w->boxes[k]);
    }
}

static void solve_collisions(World *w) {
    const int iters = 2;   /* fluid-fluid is handled by solve_fluid; 2 iters enough for solids */
    for (int it = 0; it < iters; it++) {
        for (int i = 0; i < w->bodyCount; i++) {
            Body *A = &w->bodies[i];
            if (A->tagged == 3) continue;   /* fluid vs non-fluid only here */
            for (int j = i + 1; j < w->bodyCount; j++) {
                Body *B = &w->bodies[j];
                float dx = B->pos.x - A->pos.x;
                float dy = B->pos.y - A->pos.y;
                float dz = B->pos.z - A->pos.z;
                float r = A->radius + B->radius;
                float d2 = dx*dx + dy*dy + dz*dz;
                if (d2 >= r * r || d2 < 1e-8f) continue;
                float d = sqrtf(d2);
                float pen = r - d;
                float nx = dx / d, ny = dy / d, nz = dz / d;
                float wA = A->invMass, wB = B->invMass, wSum = wA + wB;
                if (wSum == 0.0f) continue;
                A->pos.x -= nx * pen * (wA / wSum);
                A->pos.y -= ny * pen * (wA / wSum);
                A->pos.z -= nz * pen * (wA / wSum);
                B->pos.x += nx * pen * (wB / wSum);
                B->pos.y += ny * pen * (wB / wSum);
                B->pos.z += nz * pen * (wB / wSum);

                /* restitution impulse only on first pass, else it multiplies 16x */
                if (it == 0) {
                    float rvn = (B->pos.x - B->prev.x - A->pos.x + A->prev.x) * nx
                              + (B->pos.y - B->prev.y - A->pos.y + A->prev.y) * ny
                              + (B->pos.z - B->prev.z - A->pos.z + A->prev.z) * nz;
                    if (rvn < 0.0f) {
                        float e = A->restitution * B->restitution;
                        float jimp = -(1.0f + e) * rvn / wSum * 0.5f;
                        A->prev.x -= nx * jimp * wA; A->prev.y -= ny * jimp * wA; A->prev.z -= nz * jimp * wA;
                        B->prev.x += nx * jimp * wB; B->prev.y += ny * jimp * wB; B->prev.z += nz * jimp * wB;
                    }
                }
            }
        }

        for (int i = 0; i < w->bodyCount; i++) {
            Body *b = &w->bodies[i];
            float e = b->restitution;
            if (b->pos.y - b->radius < 0.0f) {
                b->pos.y = b->radius;
                float vy = b->pos.y - b->prev.y;
                b->prev.y = b->pos.y + vy * e;
                b->prev.x += (b->pos.x - b->prev.x) * 0.12f;
                b->prev.z += (b->pos.z - b->prev.z) * 0.12f;
            }
            if (b->pos.x + b->radius >  ARENA) { b->pos.x =  ARENA - b->radius; }
            if (b->pos.x - b->radius < -ARENA) { b->pos.x = -ARENA + b->radius; }
            if (b->pos.z + b->radius >  ARENA) { b->pos.z =  ARENA - b->radius; }
            if (b->pos.z - b->radius < -ARENA) { b->pos.z = -ARENA + b->radius; }
        }
    }
}

/* deterministic periodic spawner: drops a new object every 120 steps */
static void spawner_step(World *w) {
    if (w->stepCounter == 0 || w->stepCounter % 120 != 0) return;
    if (w->bodyCount + 12 >= MAX_BODIES || w->conCount + 20 >= MAX_CONS) return;
    int kind = (int)(rng_next(w) % 3);
    float x = rng_range(w, -10, 10), z = rng_range(w, -10, 10);
    float y = rng_range(w, 18.0f, 26.0f);
    if (kind == 0)
        spawn_tetra(w, x, y, z, rng_range(w, 0.5f, 0.9f), rng_range(w, 0, 6.28f));
    else if (kind == 1)
        spawn_cube(w, x, y, z, rng_range(w, 0.4f, 0.8f), rng_range(w, 0.4f, 0.8f), rng_range(w, 0.4f, 0.8f),
                   rng_range(w, 0, 6.28f));
    else
        spawn_ragdoll(w, x, y, z, rng_range(w, 0.9f, 1.2f));
}

/* settle pass: kill micro-jitter velocities on ragdoll parts */
static void settle_pass(World *w) {
    const float vThresh = 0.0012f;   /* must stay below gravity step (9/3600 = 0.0025) */
    for (int i = 0; i < w->bodyCount; i++) {
        Body *b = &w->bodies[i];
        if (b->invMass == 0.0f) continue;
        float vx = b->pos.x - b->prev.x;
        float vy = b->pos.y - b->prev.y;
        float vz = b->pos.z - b->prev.z;
        float sp2 = vx*vx + vy*vy + vz*vz;
        if (b->tagged == 3) continue;   /* fluid never settles */
        if (b->tagged == 1 && sp2 < vThresh * vThresh) {
            /* snap prev to pos: zero velocity entirely */
            b->prev = b->pos;
        } else if (sp2 < 1e-8f) {
            b->prev = b->pos;
        }
    }
}

/* fluid: block of verlet water particles (dam-break drop) */
static void spawn_water(World *w, float x0, float y0, float z0, int nx, int ny, int nz, float spacing) {
    for (int i = 0; i < nx; i++)
        for (int j = 0; j < ny; j++)
            for (int k = 0; k < nz; k++) {
                if (w->bodyCount >= MAX_BODIES) return;
                /* tiny deterministic jitter breaks perfect grid symmetry */
                float jx = rng_range(w, -0.02f, 0.02f);
                float jz = rng_range(w, -0.02f, 0.02f);
                add_body(w, x0 + i * spacing + jx, y0 + j * spacing, z0 + k * spacing + jz,
                         0.14f, 1.0f, 0.0f, 3);
            }
}

/* fluid-fluid interaction with spatial hash grid: pressure repulsion + viscosity.
 * O(n) instead of O(n^2): particles bucketed by cell = rest spacing. */
#define FLUID_REST 0.30f
#define FLUID_CELL FLUID_REST

#include "fluid_fast.h"

/* ---- realistic water behavior ----
 * Water surface height field for buoyancy/splashes: coarse grid average of
 * near-surface particle heights (deterministic fixed-order accumulation).
 */


/* water level: basin full height approx = fluid spawn volume; tuned to spawn block */
#define WATER_LEVEL 2.2f

/* estimate water height at (x,z): from wave grid if that cell is wet, else -1 (dry) */
static float water_height_at(World *w, float x, float z) {
    int gx = (int)((x + ARENA) / WAV_CELL);
    int gz = (int)((z + ARENA) / WAV_CELL);
    if (gx < 0 || gz < 0 || gx >= WAV_N || gz >= WAV_N) return -1.0f;
    WaveCell *c = &w->waveGrid[gx][gz];
    if (!c->wet) return -1.0f;
    return c->h;
}

/* rebuild wave height field: average height of fluid particles that have
 * few neighbors above them (surface particles), with simple vertical wave
 * propagation for ripples. Deterministic: fixed order, no randomness. */
static void water_surface_update(World *w) {
    WaveCell (*waveGrid)[WAV_N] = w->waveGrid;
    /* decay previous velocities + propagate to neighbors (ripple wave equation) */
    for (int i = 0; i < WAV_N; i++)
        for (int j = 0; j < WAV_N; j++) {
            WaveCell *c = &waveGrid[i][j];
            c->count = 0;
            c->sum = 0.0f;
            c->vy *= 0.92f;
            c->foam *= 0.90f;                /* foam dissipates over time */
        }    /* sample particle heights: only particles whose density < bulk count as surface */
    for (int i = 0; i < w->bodyCount; i++) {
        Body *b = &w->bodies[i];
        if (b->tagged != 3) continue;
        if (b->density > 6.0f) continue;         /* submerged: skip */
        int gx = (int)((b->pos.x + ARENA) / WAV_CELL);
        int gz = (int)((b->pos.z + ARENA) / WAV_CELL);
        if (gx < 0 || gz < 0 || gx >= WAV_N || gz >= WAV_N) continue;
        WaveCell *c = &waveGrid[gx][gz];
        c->sum += b->pos.y;
        c->count++;
    }
    for (int i = 0; i < WAV_N; i++)
        for (int j = 0; j < WAV_N; j++) {
            WaveCell *c = &waveGrid[i][j];
            if (c->count > 0) {
                float target = c->sum / (float)c->count;
                /* wave dynamics: spring toward target + neighbor coupling */
                c->vy += (target - c->h) * 0.25f;
            } else {
                /* empty cell: relax toward neighbor average (spreads out the
                 * surface) instead of a fake global water level */
                float sum = 0.0f; int nb = 0;
                if (i > 0)          { sum += waveGrid[i-1][j].h; nb++; }
                if (i < WAV_N - 1)  { sum += waveGrid[i+1][j].h; nb++; }
                if (j > 0)          { sum += waveGrid[i][j-1].h; nb++; }
                if (j < WAV_N - 1)  { sum += waveGrid[i][j+1].h; nb++; }
                if (nb > 0) c->vy += (sum / (float)nb - c->h) * 0.10f;
            }
            if (c->vy > 0.5f) c->vy = 0.5f;
            if (c->vy < -0.5f) c->vy = -0.5f;
            c->h += c->vy;
            c->wet = (c->count > 0);
            /* clamp */
            if (c->h < 0.05f) { c->h = 0.05f; c->vy = 0.0f; }
            if (c->h > 12.0f) { c->h = 12.0f; c->vy = 0.0f; }
            /* churning surface (fast vertical motion) creates foam */
            if (c->count > 0 && c->vy < -0.06f) {
                c->foam += -c->vy * 0.5f;
                if (c->foam > 2.0f) c->foam = 2.0f;
            }
        }
}

/* buoyancy + drag for solid bodies in water (Archimedes with submersion fraction) */
static void water_buoyancy(World *w) {
    WaveCell (*waveGrid)[WAV_N] = w->waveGrid;
    for (int i = 0; i < w->bodyCount; i++) {
        Body *b = &w->bodies[i];
        if (b->tagged == 3 || b->invMass == 0.0f) continue;
        float wh = water_height_at(w, b->pos.x, b->pos.z);
        /* submersion fraction: how much of the sphere is below the surface */
        float depth = wh - (b->pos.y - b->radius);
        if (depth <= 0.0f) continue;               /* in air */
        float subm = depth / (2.0f * b->radius);
        if (subm > 1.0f) subm = 1.0f;
        /* Archimedes: F = rho_water * V * g. Water is heavy: particles have
         * mass 1/invMass=1 at radius 0.14 -> water density effectively ~ 1.0.
         * Buoyancy accel = subm * g * (rho_fluid/rho_body). We use density ratio
         * so wood-like (light) bodies float, heavy rocks sink. */
        float rhoBody = 1.0f / (b->invMass * b->radius * b->radius * b->radius * 4.18879f);
        float rhoRatio = 0.55f / rhoBody;          /* water density 0.55 in sim units */
        float buoyAcc = subm * GRAVITY * rhoRatio * 1.6f;   /* 1.6 = stability-scaled */
        if (buoyAcc > GRAVITY * 1.05f) buoyAcc = GRAVITY * 1.05f; /* barely-over-neutral cap: no rockets */
        b->pos.y += buoyAcc * FIXED_DT * FIXED_DT;
        /* water drag: quadratic-ish damping on velocity */
        Vector3 vel = Vector3Subtract(b->pos, b->prev);
        b->prev.x += vel.x * 0.10f * subm;
        b->prev.y += vel.y * 0.16f * subm;
        b->prev.z += vel.z * 0.10f * subm;
        /* splash: push water surface down where body enters fast */
        if (depth > 0.0f && depth < b->radius * 2.0f && vel.y < -0.15f) {
            int gx = (int)((b->pos.x + ARENA) / WAV_CELL);
            int gz = (int)((b->pos.z + ARENA) / WAV_CELL);
            if (gx >= 0 && gz >= 0 && gx < WAV_N && gz < WAV_N) {
                WaveCell *c = &waveGrid[gx][gz];
                c->vy += vel.y * subm * 0.8f;      /* surface dips */
                c->foam += (-vel.y) * subm;        /* churn -> foam marker */
                if (c->foam > 2.0f) c->foam = 2.0f;
            }
        }
    }
}

/* surface tension: pull near-surface fluid particles toward local surface plane.
 * Cheap curvature proxy: particles with low density get cohesion toward the
 * average of neighbors (clustering of surface into smooth sheets). */
static void water_surface_tension(World *w) {
    const float h = FL_H;
    for (int fi = 0; fi < fg.nFluid; fi++) {
        int i = fg.fluidIdx[fi];
        Body *A = &w->bodies[i];
        if (A->density >= 6.0f) continue;          /* submerged: no tension */
        int cx = (int)((A->pos.x - fg.x0) / FL_CELL) + 1;
        int cy = (int)((A->pos.y - fg.y0) / FL_CELL) + 1;
        int cz = (int)((A->pos.z - fg.z0) / FL_CELL) + 1;
        Vec3 acc = {0, 0, 0};                       /* linalg vec3 accumulator */
        int nb = 0;
        for (int d = 0; d < FL_NDOFF; d++) {
            int s, e;
            fluid_cell_range(cx + fl_doff[d][0], cy + fl_doff[d][1], cz + fl_doff[d][2], &s, &e);
            for (int k = s; k < e; k++) {
                int j = fg.entries[k];
                if (j == i) continue;
                Body *B = &w->bodies[j];
                Vec3 dv = vec3_sub(vec3(B->pos.x, B->pos.y, B->pos.z),
                                   vec3(A->pos.x, A->pos.y, A->pos.z));
                float r2 = vec3_len2(dv);
                if (r2 >= h*h || r2 < 1e-12f) continue;
                acc = vec3_add(acc, dv);
                nb++;
            }
        }
        if (nb < 2) continue;
        /* cohesion: accelerate toward neighbor centroid, stronger for fewer neighbors */
        Vec3 pull = vec3_mul(acc, 0.015f / (float)nb);
        A->pos.x += pull.x; A->pos.y += pull.y; A->pos.z += pull.z;
        /* surface smoothing: damp horizontal velocity slightly on surface */
        A->prev.x += (A->pos.x - A->prev.x) * 0.02f;
        A->prev.z += (A->pos.z - A->prev.z) * 0.02f;
    }
}

/* vorticity confinement: detect curl of velocity field, add swirl force.
 * Computes coarse curl from particle velocities, re-applies as perpendicular
 * force. Keeps water lively (whirlpools) without external randomness. */
static void water_vorticity(World *w) {
    /* curl estimate per particle from central differences of neighbor velocities */
    for (int fi = 0; fi < fg.nFluid; fi++) {
        int i = fg.fluidIdx[fi];
        Body *A = &w->bodies[i];
        int cx = (int)((A->pos.x - fg.x0) / FL_CELL) + 1;
        int cy = (int)((A->pos.y - fg.y0) / FL_CELL) + 1;
        int cz = (int)((A->pos.z - fg.z0) / FL_CELL) + 1;
        Vec3 va = vec3(A->pos.x - A->prev.x, A->pos.y - A->prev.y, A->pos.z - A->prev.z);
        Vec3 curl = {0, 0, 0};
        int nb = 0;
        for (int d = 0; d < FL_NDOFF; d++) {
            int s, e;
            fluid_cell_range(cx + fl_doff[d][0], cy + fl_doff[d][1], cz + fl_doff[d][2], &s, &e);
            for (int k = s; k < e; k++) {
                int j = fg.entries[k];
                if (j == i) continue;
                Body *B = &w->bodies[j];
                Vec3 vb = vec3(B->pos.x - B->prev.x, B->pos.y - B->prev.y, B->pos.z - B->prev.z);
                Vec3 dv = vec3_sub(vec3(B->pos.x, B->pos.y, B->pos.z),
                                   vec3(A->pos.x, A->pos.y, A->pos.z));
                float r2 = vec3_len2(dv);
                if (r2 >= FL_H2 || r2 < 1e-12f) continue;
                /* curl contribution: cross(dv, vb - va) */
                curl = vec3_add(curl, vec3_cross(dv, vec3_sub(vb, va)));
                nb++;
            }
        }
        if (nb < 3) continue;
        /* apply as velocity nudge, clamped so it can never exceed the
         * gravity step (0.0025): pure stabilizer, never an energy source */
        curl = vec3_mul(curl, 1.0f / (float)nb);
        Vec3 n = vec3_norm(curl);
        Vec3 f = vec3_cross(n, va);          /* N x v: perpendicular swirl */
        float fl = vec3_len(f);
        if (fl > 0.0005f) f = vec3_mul(f, 0.0005f / fl);
        A->prev.x -= f.x; A->prev.y -= f.y; A->prev.z -= f.z;
    }
}

/* invisible container walls for fluid: keeps water in the arena basin */
static void fluid_container(World *w) {
    const float wall = ARENA - 0.5f;
    const float maxV = 0.08f;   /* max per-step fluid speed: prevents wall climb/jet instability */
    for (int i = 0; i < w->bodyCount; i++) {
        Body *b = &w->bodies[i];
        if (b->tagged != 3 || b->invMass == 0.0f) continue;
        /* velocity clamp (prev-nudge): kills tunneling + pressure spikes */
        float vx = b->pos.x - b->prev.x;
        float vy = b->pos.y - b->prev.y;
        float vz = b->pos.z - b->prev.z;
        float sp2 = vx*vx + vy*vy + vz*vz;
        if (sp2 > maxV * maxV) {
            float s = maxV / sqrtf(sp2);
            b->prev.x = b->pos.x - vx * s;
            b->prev.y = b->pos.y - vy * s;
            b->prev.z = b->pos.z - vz * s;
        }
        if (b->pos.x >  wall - b->radius) b->pos.x =  wall - b->radius;
        if (b->pos.x < -wall + b->radius) b->pos.x = -wall + b->radius;
        if (b->pos.z >  wall - b->radius) b->pos.z =  wall - b->radius;
        if (b->pos.z < -wall + b->radius) b->pos.z = -wall + b->radius;
        if (b->pos.y < b->radius) b->pos.y = b->radius;   /* basin floor */
    }
}

static void world_step(World *w) {
    apply_gravity(w);
    water_buoyancy(w);       /* Archimedes on solids before integration */
    verlet_integrate(w);
    fluid_grid_build(w);     /* rebuild spatial hash each step (deterministic: fixed iteration order) */
    solve_fluid(w);
    water_surface_tension(w);/* cohesion on near-surface particles (uses density cache) */
    water_vorticity(w);      /* swirl preservation */
    water_surface_update(w); /* wave height field for buoyancy + splashes */
    solve_constraints(w);
    solve_angular(w);
    solve_constraints(w);
    collide_statics(w);
    solve_collisions(w);
    fluid_container(w);
    settle_pass(w);
    w->stepCounter++;
    spawner_step(w);
}

#endif /* VERLET3D_H */













