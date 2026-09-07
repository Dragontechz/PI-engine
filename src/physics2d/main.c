/* Deterministic Verlet physics engine rendered with raylib.
 *
 * Determinism rules:
 *   - Fixed timestep (dt never varies with frame time)
 *   - No per-frame randomness: a seeded PRNG (xorshift64) drives everything
 *   - Solver iterations are fixed and ordered (same order every step)
 *   - No floating point env dependence (no fast-math, no FMA reordering)
 *
 * Shapes:
 *   - Balls, rigid triangles (3 particles), rigid boxes (4 particles + braces)
 *   - Ragdolls: head/torso/arms/legs made of particles + joint sticks
 *   - Static triangle blocks: level geometry, particle-vs-edge collision
 */

#include "raylib.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_BODIES 384
#define MAX_CONS 1024
#define MAX_TRI 32
#define SOLVER_ITERS 12
#define FIXED_DT (1.0f / 30.0f)   /* quarter-speed sim: 30 steps/s */
#define GRAVITY 450.0f
#define MAX_ACCUM 0.25f
#define SW 1280
#define SH 720

typedef struct {
    Vector2 pos;
    Vector2 prev;      /* verlet: prev-pos encodes velocity */
    Vector2 acc;
    float radius;
    float invMass;
    float restitution;
    int tagged;        /* 0 = ball, 1 = ragdoll, 2 = rigid-frame particle */
} Body;

typedef struct {
    int a, b;
    float restLength;
    float stiffness;
    int thick;         /* draw as fat limb (ragdoll) */
    int hidden;        /* internal brace: don't draw */
} Constraint;

typedef struct { int a, b, c; Color col; } Tri;   /* rigid filled shape */

typedef struct { float x1, y1, x2, y2; } Segment; /* static block edge */

typedef struct {
    Body bodies[MAX_BODIES];
    int bodyCount;

    Constraint cons[MAX_CONS];
    int conCount;

    Tri tris[MAX_TRI];
    int triCount;

    Segment segs[MAX_TRI * 3];
    int segCount;

    unsigned long long rngState;
    unsigned long long stepCounter;
    double accumulator;
} World;

/* ---------- deterministic PRNG (xorshift64*) ---------- */
static unsigned long long rng_next(World *w) {
    unsigned long long x = w->rngState;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
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

/* ---------- builders ---------- */
static int add_body(World *w, float x, float y, float r, float invMass, float rest, int tagged) {
    if (w->bodyCount >= MAX_BODIES) return -1;
    Body *b = &w->bodies[w->bodyCount];
    b->pos = (Vector2){x, y};
    b->prev = b->pos;
    b->acc = (Vector2){0, 0};
    b->radius = r;
    b->invMass = invMass;
    b->restitution = rest;
    b->tagged = tagged;
    return w->bodyCount++;
}

static void add_con(World *w, int a, int b, float stiff, int thick, int hidden) {
    if (w->conCount >= MAX_CONS) return;
    Constraint *c = &w->cons[w->conCount++];
    c->a = a; c->b = b;
    float dx = w->bodies[b].pos.x - w->bodies[a].pos.x;
    float dy = w->bodies[b].pos.y - w->bodies[a].pos.y;
    c->restLength = sqrtf(dx * dx + dy * dy);
    c->stiffness = stiff;
    c->thick = thick;
    c->hidden = hidden;
}

/* rigid triangle body: 3 particles + 3 edges (triangle is rigid by itself) */
static void spawn_triangle(World *w, float cx, float cy, float size, float angle) {
    int idx[3];
    for (int i = 0; i < 3; i++) {
        float a = angle + (float)i / 3.0f * 6.2831853f;
        idx[i] = add_body(w, cx + cosf(a) * size, cy + sinf(a) * size,
                          5.0f, 1.0f, 0.4f, 2);
    }
    for (int i = 0; i < 3; i++)
        add_con(w, idx[i], idx[(i + 1) % 3], 1.0f, 0, 0);
    if (w->triCount < MAX_TRI)
        w->tris[w->triCount++] = (Tri){idx[0], idx[1], idx[2], (Color){150, 250, 150, 255}};
}

/* rigid box ("exoskeleton"): 4 particles, 4 edges + 2 diagonal braces */
static void spawn_box(World *w, float cx, float cy, float hw, float hh, float angle) {
    int idx[4];
    Vector2 corners[4] = {{-hw,-hh},{hw,-hh},{hw,hh},{-hw,hh}};
    float ca = cosf(angle), sa = sinf(angle);
    for (int i = 0; i < 4; i++) {
        float x = cx + corners[i].x * ca - corners[i].y * sa;
        float y = cy + corners[i].x * sa + corners[i].y * ca;
        idx[i] = add_body(w, x, y, 5.0f, 1.0f, 0.4f, 2);
    }
    for (int i = 0; i < 4; i++)
        add_con(w, idx[i], idx[(i + 1) % 4], 1.0f, 0, 0);
    add_con(w, idx[0], idx[2], 1.0f, 0, 1);   /* braces hidden */
    add_con(w, idx[1], idx[3], 1.0f, 0, 1);
    if (w->triCount < MAX_TRI)
        w->tris[w->triCount++] = (Tri){idx[0], idx[1], idx[2], (Color){250, 180, 80, 255}};
}

/* classic verlet ragdoll: head, shoulders, hips, arms, legs */
static void spawn_ragdoll(World *w, float x, float y, float scale) {
    float s = scale;
    int head  = add_body(w, x, y, 12 * s, 1.0f, 0.02f, 1);
    int shoul = add_body(w, x, y + 22 * s, 5 * s, 1.0f, 0.02f, 1);
    int hip   = add_body(w, x, y + 52 * s, 5 * s, 1.0f, 0.02f, 1);
    int elbL  = add_body(w, x - 18 * s, y + 36 * s, 4 * s, 1.0f, 0.02f, 1);
    int elbR  = add_body(w, x + 18 * s, y + 36 * s, 4 * s, 1.0f, 0.02f, 1);
    int hndL  = add_body(w, x - 24 * s, y + 58 * s, 4 * s, 1.0f, 0.02f, 1);
    int hndR  = add_body(w, x + 24 * s, y + 58 * s, 4 * s, 1.0f, 0.02f, 1);
    int kneL  = add_body(w, x - 10 * s, y + 82 * s, 4 * s, 1.0f, 0.02f, 1);
    int kneR  = add_body(w, x + 10 * s, y + 82 * s, 4 * s, 1.0f, 0.02f, 1);
    int fotL  = add_body(w, x - 12 * s, y + 108 * s, 4 * s, 1.0f, 0.02f, 1);
    int fotR  = add_body(w, x + 12 * s, y + 108 * s, 4 * s, 1.0f, 0.02f, 1);

    add_con(w, head,  shoul, 1.0f, 1, 0);   /* neck */
    add_con(w, shoul, hip,   1.0f, 1, 0);   /* spine */
    add_con(w, shoul, elbL,  1.0f, 1, 0);
    add_con(w, elbL,  hndL,  1.0f, 1, 0);
    add_con(w, shoul, elbR,  1.0f, 1, 0);
    add_con(w, elbR,  hndR,  1.0f, 1, 0);
    add_con(w, hip,   kneL,  1.0f, 1, 0);
    add_con(w, kneL,  fotL,  1.0f, 1, 0);
    add_con(w, hip,   kneR,  1.0f, 1, 0);
    add_con(w, kneR,  fotR,  1.0f, 1, 0);

    /* soft cross-braces keep the torso from folding */
    add_con(w, head,  hip,   0.2f, 0, 1);
    add_con(w, elbL,  hndR,  0.05f, 0, 1);
}

/* static triangle block: level geometry with collision edges */
static void add_static_tri(World *w, float x1, float y1, float x2, float y2, float x3, float y3) {
    if (w->segCount + 3 > MAX_TRI * 3) return;
    w->segs[w->segCount++] = (Segment){x1, y1, x2, y2};
    w->segs[w->segCount++] = (Segment){x2, y2, x3, y3};
    w->segs[w->segCount++] = (Segment){x3, y3, x1, y1};
    if (w->triCount < MAX_TRI) {
        /* render-only triangle; body indices unused (-1) */
        w->tris[w->triCount++] = (Tri){-1, -1, -1, (Color){70, 90, 110, 255}};
        /* stash vertices in bodies array would pollute physics; instead draw from segs */
        w->triCount--; /* draw statics from segs directly instead */
    }
}

/* ---------- world setup (pure function of seed => replayable) ---------- */
static void world_reset(World *w, unsigned long long seed) {
    memset(w, 0, sizeof(*w));
    rng_seed(w, seed);

    /* static triangle blocks (same every seed) */
    add_static_tri(w, 100, 720, 380, 720, 240, 560);    /* left ramp */
    add_static_tri(w, 1280, 720, 900, 720, 1090, 560);  /* right ramp */
    add_static_tri(w, 520, 420, 760, 420, 640, 540);    /* center pyramid */

    /* rigid triangles falling from RNG positions */
    for (int i = 0; i < 3; i++)
        spawn_triangle(w, rng_range(w, 200, 1080), rng_range(w, -150, 0),
                       rng_range(w, 22, 36), rng_range(w, 0, 6.28f));

    /* rigid boxes */
    for (int i = 0; i < 3; i++)
        spawn_box(w, rng_range(w, 200, 1080), rng_range(w, -250, -100),
                  rng_range(w, 20, 34), rng_range(w, 14, 26), rng_range(w, 0, 6.28f));

    /* ragdolls */
    spawn_ragdoll(w, rng_range(w, 300, 500), rng_range(w, -300, -150), 1.0f);
    spawn_ragdoll(w, rng_range(w, 780, 980), rng_range(w, -400, -250), 1.1f);

    /* a pyramid of balls */
    const int rows = 5;
    float startX = 640.0f, startY = 100.0f, spacing = 24.0f;
    for (int r = 0; r < rows; r++)
        for (int c = 0; c <= r; c++)
            add_body(w, startX - r * spacing * 0.5f + c * spacing, startY + r * spacing,
                     10.0f, 1.0f, 0.2f, 0);

    /* rope */
    const int chainN = 8;
    int chainStart = w->bodyCount;
    for (int i = 0; i < chainN; i++) {
        int id = add_body(w, 200.0f, 60.0f + i * 28.0f, 6.0f, i == 0 ? 0.0f : 1.0f, 0.1f, 0);
        if (i > 0) add_con(w, id - 1, id, 1.0f, 0, 0);
    }
    (void)chainStart;

    w->accumulator = 0.0;
}

/* ---------- integration ---------- */
static void apply_gravity(World *w) {
    for (int i = 0; i < w->bodyCount; i++) {
        Body *b = &w->bodies[i];
        if (b->invMass == 0.0f) continue;
        b->acc.y += GRAVITY;
    }
}

static void verlet_integrate(World *w) {
    const float dt = FIXED_DT;
    for (int i = 0; i < w->bodyCount; i++) {
        Body *b = &w->bodies[i];
        if (b->invMass == 0.0f) { b->acc = (Vector2){0, 0}; continue; }
        Vector2 vel = {b->pos.x - b->prev.x, b->pos.y - b->prev.y};
        const float damp = 0.998f;   /* more air drag => less floaty, more grounded */
        Vector2 next = {
            b->pos.x + vel.x * damp + b->acc.x * dt * dt,
            b->pos.y + vel.y * damp + b->acc.y * dt * dt
        };
        b->prev = b->pos;
        b->pos = next;
        b->acc = (Vector2){0, 0};
    }
}

/* ---------- constraints ---------- */
static void solve_constraints(World *w) {
    for (int it = 0; it < SOLVER_ITERS; it++) {
        for (int i = 0; i < w->conCount; i++) {
            Constraint *c = &w->cons[i];
            Body *A = &w->bodies[c->a];
            Body *B = &w->bodies[c->b];
            float dx = B->pos.x - A->pos.x;
            float dy = B->pos.y - A->pos.y;
            float d2 = dx * dx + dy * dy;
            if (d2 < 1e-8f) continue;
            float d = sqrtf(d2);
            float diff = (d - c->restLength) / d * c->stiffness;
            float wA = A->invMass, wB = B->invMass;
            float wSum = wA + wB;
            if (wSum == 0.0f) continue;
            A->pos.x += dx * diff * (wA / wSum);
            A->pos.y += dy * diff * (wA / wSum);
            B->pos.x -= dx * diff * (wB / wSum);
            B->pos.y -= dy * diff * (wB / wSum);
        }
    }
}

/* ---------- particle vs static block edges ---------- */
static void collide_statics(World *w) {
    for (int s = 0; s < w->segCount; s++) {
        Segment *sg = &w->segs[s];
        float ex = sg->x2 - sg->x1, ey = sg->y2 - sg->y1;
        float len2 = ex * ex + ey * ey;
        if (len2 < 1e-8f) continue;
        for (int i = 0; i < w->bodyCount; i++) {
            Body *b = &w->bodies[i];
            if (b->invMass == 0.0f) continue;
            float t = ((b->pos.x - sg->x1) * ex + (b->pos.y - sg->y1) * ey) / len2;
            if (t < 0.0f) t = 0.0f;
            if (t > 1.0f) t = 1.0f;
            float cx = sg->x1 + ex * t, cy = sg->y1 + ey * t;
            float dx = b->pos.x - cx, dy = b->pos.y - cy;
            float d2 = dx * dx + dy * dy;
            float r = b->radius + 2.0f;
            if (d2 >= r * r || d2 < 1e-8f) continue;
            float d = sqrtf(d2);
            float nx = dx / d, ny = dy / d;
            float pen = r - d;
            b->pos.x += nx * pen;
            b->pos.y += ny * pen;
            /* friction: pull prev toward contact motion (kills sliding) */
            b->prev.x += nx * pen * 0.9f;
            b->prev.y += ny * pen * 0.9f;
            /* tangential ground friction */
            {
                float vx = b->pos.x - b->prev.x, vy = b->pos.y - b->prev.y;
                float vn = vx * nx + vy * ny;
                float tx = vx - vn * nx, ty = vy - vn * ny;
                b->prev.x -= tx * 0.15f;   /* eat 15% of tangential velocity */
                b->prev.y -= ty * 0.15f;
            }
        }
    }
}

/* ---------- collisions (deterministic order: sorted index pairs) ---------- */
static void solve_collisions(World *w) {
    for (int it = 0; it < SOLVER_ITERS; it++) {
        for (int i = 0; i < w->bodyCount; i++) {
            for (int j = i + 1; j < w->bodyCount; j++) {
                Body *A = &w->bodies[i];
                Body *B = &w->bodies[j];
                float dx = B->pos.x - A->pos.x;
                float dy = B->pos.y - A->pos.y;
                float r = A->radius + B->radius;
                float d2 = dx * dx + dy * dy;
                if (d2 >= r * r || d2 < 1e-8f) continue;
                float d = sqrtf(d2);
                float pen = r - d;
                float nx = dx / d, ny = dy / d;
                float wA = A->invMass, wB = B->invMass;
                float wSum = wA + wB;
                if (wSum == 0.0f) continue;
                A->pos.x -= nx * pen * (wA / wSum);
                A->pos.y -= ny * pen * (wA / wSum);
                B->pos.x += nx * pen * (wB / wSum);
                B->pos.y += ny * pen * (wB / wSum);

                float vax = A->pos.x - A->prev.x, vay = A->pos.y - A->prev.y;
                float vbx = B->pos.x - B->prev.x, vby = B->pos.y - B->prev.y;
                float rvn = (vbx - vax) * nx + (vby - vay) * ny;
                if (rvn < 0.0f) {
                    float e = A->restitution * B->restitution;
                    float jimp = -(1.0f + e) * rvn / wSum;
                    float jx = nx * jimp * 0.5f, jy = ny * jimp * 0.5f;
                    A->prev.x -= jx * wA; A->prev.y -= jy * wA;
                    B->prev.x += jx * wB; B->prev.y += jy * wB;
                }
            }
        }

        /* walls */
        for (int i = 0; i < w->bodyCount; i++) {
            Body *b = &w->bodies[i];
            float e = b->restitution;
            if (b->pos.y + b->radius > (float)SH) {
                b->pos.y = (float)SH - b->radius;
                float vy = b->pos.y - b->prev.y;
                b->prev.y = b->pos.y + vy * e;
            }
            if (b->pos.y - b->radius < -2000.0f) { /* deep pit kill-floor safety */
                b->pos.y = -2000.0f + b->radius;
            }
            if (b->pos.x + b->radius > (float)SW) {
                b->pos.x = (float)SW - b->radius;
                float vx = b->pos.x - b->prev.x;
                b->prev.x = b->pos.x + vx * e;
            }
            if (b->pos.x - b->radius < 0.0f) {
                b->pos.x = b->radius;
                float vx = b->pos.x - b->prev.x;
                b->prev.x = b->pos.x + vx * e;
            }
        }
    }
}

/* ---------- one fixed physics step (fully deterministic) ---------- */
static void world_step(World *w) {
    apply_gravity(w);
    verlet_integrate(w);
    solve_constraints(w);
    collide_statics(w);
    solve_collisions(w);
    w->stepCounter++;
}

/* mouse grab */
static int grabbed = -1;
static Vector2 grabTarget;

static void apply_mouse(World *w) {
    if (grabbed < 0 || grabbed >= w->bodyCount) return;
    Body *b = &w->bodies[grabbed];
    Vector2 vel = {b->pos.x - b->prev.x, b->pos.y - b->prev.y};
    Vector2 want = {grabTarget.x - b->pos.x, grabTarget.y - b->pos.y};
    b->prev.x = b->pos.x - vel.x * 0.5f - want.x * 0.15f;
    b->prev.y = b->pos.y - vel.y * 0.5f - want.y * 0.15f;
}

int main(void) {
    InitWindow(SW, SH, "Deterministic Verlet Physics - raylib");
    SetTargetFPS(60);

    unsigned long long seed = 42;
    static World world;
    world_reset(&world, seed);

    char status[256];

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_R)) { world_reset(&world, seed); grabbed = -1; }
        if (IsKeyPressed(KEY_N)) { seed++; world_reset(&world, seed); grabbed = -1; }

        Vector2 mouse = GetMousePosition();
        if (IsMouseButtonPressed(MOUSE_LEFT_BUTTON)) {
            float best = 1e9f;
            grabbed = -1;
            for (int i = 0; i < world.bodyCount; i++) {
                Body *b = &world.bodies[i];
                float dx = mouse.x - b->pos.x, dy = mouse.y - b->pos.y;
                float d = dx * dx + dy * dy;
                if (d < best && d < 60.0f * 60.0f) { best = d; grabbed = i; }
            }
            grabTarget = mouse;
        }
        if (IsMouseButtonReleased(MOUSE_LEFT_BUTTON)) grabbed = -1;
        grabTarget = mouse;

        double frame = GetFrameTime();
        if (frame > MAX_ACCUM) frame = MAX_ACCUM;
        world.accumulator += frame;
        while (world.accumulator >= (double)FIXED_DT) {
            apply_mouse(&world);
            world_step(&world);
            world.accumulator -= (double)FIXED_DT;
        }

        snprintf(status, sizeof(status),
                 "seed: %llu   steps: %llu   bodies: %d   [R] reset  [N] new seed  LMB drag",
                 seed, world.stepCounter, world.bodyCount);

        BeginDrawing();
        ClearBackground((Color){18, 18, 24, 255});

        /* static blocks (from segments, filled triangles via fan) */
        for (int s = 0; s + 2 < world.segCount; s += 3) {
            Segment *a = &world.segs[s], *c = &world.segs[s + 2];
            DrawTriangle((Vector2){a->x1, a->y1},
                         (Vector2){a->x2, a->y2},
                         (Vector2){c->x2, c->y2},
                         (Color){70, 90, 110, 255});
            DrawLineEx((Vector2){a->x1, a->y1}, (Vector2){a->x2, a->y2}, 2.0f, (Color){110, 140, 170, 255});
            DrawLineEx((Vector2){a->x2, a->y2}, (Vector2){c->x2, c->y2}, 2.0f, (Color){110, 140, 170, 255});
            DrawLineEx((Vector2){c->x2, c->y2}, (Vector2){a->x1, a->y1}, 2.0f, (Color){110, 140, 170, 255});
        }

        /* rigid filled triangles / boxes */
        for (int i = 0; i < world.triCount; i++) {
            Tri *t = &world.tris[i];
            if (t->a < 0) continue;
            Vector2 p0 = world.bodies[t->a].pos;
            Vector2 p1 = world.bodies[t->b].pos;
            Vector2 p2 = world.bodies[t->c].pos;
            DrawTriangle(p0, p1, p2, t->col);
            DrawTriangleLines(p0, p1, p2, (Color){255, 255, 255, 80});
        }

        /* constraints (sticks / limbs) */
        for (int i = 0; i < world.conCount; i++) {
            Constraint *c = &world.cons[i];
            if (c->hidden) continue;
            Body *A = &world.bodies[c->a], *B = &world.bodies[c->b];
            if (c->thick)
                DrawLineEx(A->pos, B->pos, 7.0f, (Color){240, 190, 150, 255});
            else
                DrawLineEx(A->pos, B->pos, 2.0f, (Color){120, 120, 160, 255});
        }

        /* bodies */
        for (int i = 0; i < world.bodyCount; i++) {
            Body *b = &world.bodies[i];
            Color col;
            if (b->invMass == 0.0f)      col = (Color){230, 90, 90, 255};
            else if (b->tagged == 1)     col = (Color){250, 210, 180, 255}; /* ragdoll joints */
            else if (b->tagged == 2)     col = (Color){250, 180, 80, 255};  /* frame corners */
            else                         col = (Color){90, 200, 250, 255};
            if (b->tagged == 1)
                DrawCircleV(b->pos, b->radius, col);   /* ragdoll joints blend into limbs */
            else
                DrawCircleV(b->pos, b->radius * 0.6f, col);
        }

        if (grabbed >= 0)
            DrawCircleLines((int)grabTarget.x, (int)grabTarget.y, 20.0f, (Color){255, 220, 90, 255});

        DrawText(status, 12, 12, 18, (Color){200, 200, 210, 255});
        EndDrawing();
    }

    CloseWindow();
    return 0;
}
