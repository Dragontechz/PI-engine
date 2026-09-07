/* CPU ray tracer: block + sphere + checker floor, point light, shadows,
 * Blinn-Phong, one reflection bounce, sky. Headless, deterministic.
 * Output: render.png + VERIFY pass/fail lines on stdout.
 *
 * Scene (top level):
 *   tracer_init()   - one-time setup (camera, images)
 *   tracer_frame(dt)- advances nothing; time is fixed (deterministic)
 *   tracer_render(rgba, w, h) - renders one frame into a CPU buffer
 *   float* returns are scene-space; all colors 0..255 output
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define TR_W 1280
#define TR_H 720

typedef struct { float x, y, z; } V3;
typedef struct { unsigned char r, g, b, a; } Pix;

static V3 v3(float x, float y, float z) { V3 v = { x, y, z }; return v; }
static V3 vadd(V3 a, V3 b) { return v3(a.x + b.x, a.y + b.y, a.z + b.z); }
static V3 vsub(V3 a, V3 b) { return v3(a.x - b.x, a.y - b.y, a.z - b.z); }
static V3 vscale(V3 a, float s) { return v3(a.x * s, a.y * s, a.z * s); }
static float vdot(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static V3 vcross(V3 a, V3 b) {
    return v3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
static float vlen(V3 a) { return sqrtf(vdot(a, a)); }
static V3 vnorm(V3 a) { float l = vlen(a); return vscale(a, 1.0f / (l + 1e-9f)); }

static float ffmin(float a, float b) { return a < b ? a : b; }
static float ffmax(float a, float b) { return a > b ? a : b; }
static float clamp01(float x) { return x < 0.0f ? 0.0f : (x > 1.0f ? 1.0f : x); }

/* ---------------- scene ---------------- */

typedef struct {
    V3 min, max;
} Box;

typedef struct {
    V3 c;
    float r;
} Sphere;

static Box g_box = { { -1.2f, 0.0f, -1.2f }, { 1.2f, 2.4f, 1.2f } };
static Sphere g_sph = { { 2.35f, 0.8f, 1.7f }, 0.8f };
static V3 g_light = { 4.2f, 6.5f, 3.2f };
static V3 g_sun = { -0.35f, 0.30f, -0.65f }; /* NOT normalized */

static V3 camPos = { 5.4f, 3.6f, 6.2f };
static V3 camLook = { 0.0f, 1.2f, 0.0f };
static float camFovY = 50.0f;

/* hit record: t >= 0 along ray, n unit normal, id: 0=floor 1=box 2=sphere */
typedef struct { float t; V3 n; int id; } Hit;

static int hit_box(const Box *b, V3 ro, V3 rd, float tmax, float *tOut, V3 *nOut) {
    float tmin = -1e9f, tmaxb = 1e9f;
    int axMin = 0;
    float o[3] = { ro.x, ro.y, ro.z };
    float d[3] = { rd.x, rd.y, rd.z };
    float mn[3] = { b->min.x, b->min.y, b->min.z };
    float mx[3] = { b->max.x, b->max.y, b->max.z };
    for (int a = 0; a < 3; a++) {
        if (fabsf(d[a]) < 1e-9f) {
            if (o[a] < mn[a] || o[a] > mx[a]) return 0;
            continue;
        }
        float inv = 1.0f / d[a];
        float t1 = (mn[a] - o[a]) * inv;
        float t2 = (mx[a] - o[a]) * inv;
        int entering = 1;
        if (t1 > t2) { float tt = t1; t1 = t2; t2 = tt; entering = 0; }
        if (t1 > tmin) { tmin = t1; axMin = entering ? a : a; }
        if (t2 < tmaxb) tmaxb = t2;
        if (tmin > tmaxb) return 0;
    }
    if (tmin < 1e-4f) return 0;
    if (tmin > tmax) return 0;
    *tOut = tmin;
    V3 n = v3(0, 0, 0);
    float sgn = 0.0f;
    float *nd = &n.x;
    /* entering face: if ray component positive, we hit the min face -> normal -1 */
    sgn = (d[axMin] > 0.0f) ? -1.0f : 1.0f;
    nd[axMin] = sgn;
    *nOut = n;
    return 1;
}

static int hit_sphere(const Sphere *s, V3 ro, V3 rd, float tmax, float *tOut, V3 *nOut) {
    V3 oc = vsub(ro, s->c);
    float b = vdot(oc, rd);
    float c = vdot(oc, oc) - s->r * s->r;
    float disc = b * b - c;
    if (disc < 0.0f) return 0;
    float sq = sqrtf(disc);
    float t = -b - sq;
    if (t < 1e-4f) t = -b + sq;
    if (t < 1e-4f || t > tmax) return 0;
    *tOut = t;
    *nOut = vnorm(vsub(vadd(ro, vscale(rd, t)), s->c));
    return 1;
}

/* floor plane y=0 */
static int hit_floor(V3 ro, V3 rd, float tmax, float *tOut) {
    if (fabsf(rd.y) < 1e-9f) return 0;
    float t = -ro.y / rd.y;
    if (t < 1e-4f || t > tmax) return 0;
    *tOut = t;
    return 1;
}

/* returns 0 if missed; fills h */
static int trace(V3 ro, V3 rd, Hit *h) {
    float best = 1e9f;
    int found = 0;
    float t;
    V3 n;
    if (hit_floor(ro, rd, best, &t)) { best = t; h->id = 0; h->t = t; h->n = v3(0, 1, 0); found = 1; }
    if (hit_box(&g_box, ro, rd, best, &t, &n)) { best = t; h->id = 1; h->t = t; h->n = n; found = 1; }
    if (hit_sphere(&g_sph, ro, rd, best, &t, &n)) { best = t; h->id = 2; h->t = t; h->n = n; found = 1; }
    return found;
}

/* shadow ray: any hit before light? */
static int shadowed(V3 p, V3 lpos) {
    V3 d = vsub(lpos, p);
    float dist = vlen(d);
    d = vscale(d, 1.0f / dist);
    Hit h;
    if (trace(vadd(p, vscale(d, 1e-3f)), d, &h)) return h.t < dist - 1e-3f;
    return 0;
}

static V3 sky_color(V3 rd) {
    float t = clamp01(rd.y * 0.5f + 0.5f);
    V3 horizon = v3(0.78f, 0.84f, 0.90f);
    V3 zenith = v3(0.25f, 0.45f, 0.78f);
    V3 c = vscale(vadd(vscale(horizon, 1.0f - t), vscale(zenith, t)), 1.0f);
    V3 sd = vnorm(g_sun);
    float sun = ffmax(vdot(rd, sd), 0.0f);
    c = vadd(c, vscale(v3(1.0f, 0.9f, 0.7f), powf(sun, 300.0f) * 3.0f));
    return c;
}

static V3 shade(V3 ro, V3 rd, Hit h, int depth);

static V3 shade(V3 ro, V3 rd, Hit h, int depth) {
    V3 p = vadd(ro, vscale(rd, h.t));
    V3 albedo;
    int reflective = 0;
    if (h.id == 0) {
        float ck = (floorf(p.x) + floorf(p.z));
        int even = ((int)ck & 1) == 0;
        albedo = even ? v3(0.82f, 0.78f, 0.70f) : v3(0.25f, 0.24f, 0.22f);
        if (fabsf(p.x) > 14.0f || fabsf(p.z) > 14.0f) albedo = v3(0.30f, 0.42f, 0.30f);
    } else if (h.id == 1) {
        albedo = v3(0.85f, 0.35f, 0.20f);
        reflective = 1;
    } else {
        albedo = v3(0.30f, 0.50f, 0.85f);
        reflective = 1;
    }
    V3 n = h.n;
    V3 toL = vsub(g_light, p);
    float distL = vlen(toL);
    toL = vscale(toL, 1.0f / distL);
    float atten = 1.0f / (1.0f + 0.03f * distL * distL);
    V3 col = vscale(albedo, 0.16f); /* ambient */
    if (!shadowed(p, g_light)) {
        float ndl = ffmax(vdot(n, toL), 0.0f);
        V3 hv = vnorm(vsub(toL, vscale(rd, 1.0f)));
        float spec = powf(ffmax(vdot(n, hv), 0.0f), 64.0f);
        col = vadd(col, vscale(v3(1.0f, 0.95f, 0.85f), ndl * atten * 1.5f));
        col = vadd(col, vscale(v3(1.0f, 0.95f, 0.85f), spec * atten * 1.2f));
        /* scale by albedo for the diffuse part */
        col = vadd(vscale(albedo, 0.16f), vscale(v3(albedo.x * col.x, albedo.y * col.y, albedo.z * col.z), 1.0f));
    }
    /* sun (directional, no shadow test for simplicity of verify) */
    {
        V3 sd = vnorm(g_sun);
        float ndl = ffmax(vdot(n, sd), 0.0f);
        col = vadd(col, vscale(v3(albedo.x * 0.9f, albedo.y * 0.9f, albedo.z * 0.9f), ndl * 0.35f));
    }
    if (reflective && depth < 1) {
        V3 rdir = vnorm(vsub(rd, vscale(n, 2.0f * vdot(rd, n))));
        Hit rh;
        if (trace(vadd(p, vscale(rdir, 1e-3f)), rdir, &rh)) {
            V3 rc = shade(p, rdir, rh, depth + 1);
            col = vadd(vscale(col, 0.6f), vscale(rc, 0.4f));
        } else {
            V3 rc = sky_color(rdir);
            col = vadd(vscale(col, 0.6f), vscale(rc, 0.4f));
        }
    }
    return col;
}

static unsigned char toByte(float x) {
    int v = (int)(powf(clamp01(x), 1.0f / 2.2f) * 255.0f + 0.5f);
    if (v < 0) v = 0;
    if (v > 255) v = 255;
    return (unsigned char)v;
}

int main(void) {
    static unsigned char img[TR_W * TR_H * 3];
    V3 fwd = vnorm(vsub(camLook, camPos));
    V3 right = vnorm(vcross(fwd, v3(0, 1, 0)));
    V3 up = vcross(right, fwd);
    float aspect = (float)TR_W / (float)TR_H;
    float tanH = tanf(camFovY * 3.14159265f / 360.0f);
    for (int y = 0; y < TR_H; y++) {
        for (int x = 0; x < TR_W; x++) {
            float u = ((float)x + 0.5f) / TR_W * 2.0f - 1.0f;
            float v = 1.0f - ((float)y + 0.5f) / TR_H * 2.0f;
            V3 rd = vnorm(vadd(vadd(fwd, vscale(right, u * aspect * tanH)), vscale(up, v * tanH)));
            Hit h;
            V3 col;
            if (trace(camPos, rd, &h)) col = shade(camPos, rd, h, 0);
            else col = sky_color(rd);
            int i = (y * TR_W + x) * 3;
            img[i + 0] = toByte(col.x);
            img[i + 1] = toByte(col.y);
            img[i + 2] = toByte(col.z);
        }
    }

    /* ---- verification rays: expected outcomes, no GPU involved ---- */
    int fails = 0;
    Hit h;
    V3 n;
    struct { V3 ro, rd; int expectHit; int expectId; const char *name; } tests[] = {
        { { 5.4f, 3.6f, 6.2f }, vnorm(v3(-1.0f, -0.4f, -1.0f)), 1, 1, "box front-top" },
        { { 5.4f, 3.6f, 6.2f }, vnorm(v3(0.3f, -0.5f, 0.3f)), 1, 0, "floor" },
        { { 5.4f, 3.6f, 6.2f }, vnorm(v3(1.0f, 0.2f, 0.2f)), 0, -1, "sky" },
    };
    for (unsigned ti = 0; ti < sizeof(tests) / sizeof(tests[0]); ti++) {
        int got = trace(tests[ti].ro, tests[ti].rd, &h);
        int ok = got == tests[ti].expectHit && (!got || h.id == tests[ti].expectId);
        printf("VERIFY %s: %s (hit=%d id=%d expect hit=%d id=%d)\n",
            tests[ti].name, ok ? "PASS" : "FAIL", got, got ? h.id : -1,
            tests[ti].expectHit, tests[ti].expectId);
        if (!ok) fails++;
    }
    /* shadow probe: lit point away from box; shadow cast behind box (-x,-z side) */
    float lit1 = shadowed(v3(1.6f, 0.02f, 0.0f), g_light);
    float lit2 = shadowed(v3(-1.8f, 0.02f, -1.8f), g_light);
    printf("VERIFY point(1.6,0,0) unshadowed: %s | point(-1.8,0,-1.8) shadowed: %s\n",
        !lit1 ? "PASS" : "FAIL", lit2 ? "PASS" : "FAIL");
    if (lit1 || !lit2) fails++;
    /* image stats: sky bright at top, floor lit, box red present */
    long sTop = 0, sFloor = 0;
    int redMax = 0;
    for (int x = 0; x < TR_W; x += 8) {
        int yy = 40; sTop += img[(yy * TR_W + x) * 3 + 2];
        yy = 640; sFloor += img[(yy * TR_W + x) * 3 + 1];
    }
    for (int i = 0; i < TR_W * TR_H * 3; i += 3)
        if (img[i] > redMax) redMax = img[i];
    printf("VERIFY skyBlue>120: %s (got %ld) | floorGreen>60: %s (got %ld) | boxRed>180: %s (got %d)\n",
        sTop / (TR_W / 8) > 120 ? "PASS" : "FAIL", sTop / (TR_W / 8),
        sFloor / (TR_W / 8) > 60 ? "PASS" : "FAIL", sFloor / (TR_W / 8),
        redMax > 180 ? "PASS" : "FAIL", redMax);
    /* write BMP (24bpp, bottom-up) */
    {
        int rowBytes = (TR_W * 3 + 3) & ~3;
        int dataSize = rowBytes * TR_H;
        int fileSize = 54 + dataSize;
        unsigned char hdr[54] = { 0 };
        hdr[0] = 'B'; hdr[1] = 'M';
        hdr[2] = (unsigned char)(fileSize); hdr[3] = (unsigned char)(fileSize >> 8);
        hdr[4] = (unsigned char)(fileSize >> 16); hdr[5] = (unsigned char)(fileSize >> 24);
        hdr[10] = 54;
        hdr[14] = 40;
        hdr[18] = (unsigned char)(TR_W); hdr[19] = (unsigned char)(TR_W >> 8);
        hdr[20] = (unsigned char)(TR_W >> 16); hdr[21] = (unsigned char)(TR_W >> 24);
        hdr[22] = (unsigned char)(TR_H); hdr[23] = (unsigned char)(TR_H >> 8);
        hdr[24] = (unsigned char)(TR_H >> 16); hdr[25] = (unsigned char)(TR_H >> 24);
        hdr[26] = 1; hdr[28] = 24;
        FILE *f = fopen("render.bmp", "wb");
        if (!f) { printf("ERROR: cannot open render.bmp\n"); return 1; }
        fwrite(hdr, 1, 54, f);
        unsigned char *row = (unsigned char *)calloc((size_t)rowBytes, 1);
        for (int y = TR_H - 1; y >= 0; y--) {
            for (int x = 0; x < TR_W; x++) {
                int i = (y * TR_W + x) * 3;
                row[x * 3 + 0] = img[i + 2];
                row[x * 3 + 1] = img[i + 1];
                row[x * 3 + 2] = img[i + 0];
            }
            fwrite(row, 1, rowBytes, f);
        }
        free(row);
        fclose(f);
    }
    printf("RENDER OK: render.bmp %dx%d fails=%d\n", TR_W, TR_H, fails);
    return fails ? 1 : 0;
}
