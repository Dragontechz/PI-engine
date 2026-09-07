/* Deterministic 3D Verlet physics - raylib front-end.
 * Water rendering: two-pass shader water (scene -> FBO color+depth, then
 * refractive/absorptive surface composite) with CPU vertex-color fallback. */
#include "raylib.h"
#include "rlgl.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "verlet3d.h"

static int grabbed = -1;
static Vector3 grabTarget;

static void apply_mouse(World *w) {
    if (grabbed < 0 || grabbed >= w->bodyCount) return;
    Body *b = &w->bodies[grabbed];
    Vector3 vel = Vector3Subtract(b->pos, b->prev);
    Vector3 want = Vector3Subtract(grabTarget, b->pos);
    b->prev = Vector3Subtract(b->pos,
        Vector3Add(Vector3Scale(vel, 0.5f), Vector3Scale(want, 0.15f)));
}

/* ---------------- water surface mesh (wave grid) ---------------- */

static Mesh surfaceMesh;
static Color *surfCols;
static Vector3 *surfVerts;
static bool surfaceMeshInit = false;

static void surface_mesh_init(void) {
    int quads = (WAV_N - 1) * (WAV_N - 1);
    int triCount = quads * 2;
    surfaceMesh.triangleCount = triCount;
    surfaceMesh.vertexCount = triCount * 3;
    surfaceMesh.vertices = (float *)MemAlloc(sizeof(float) * triCount * 9);
    surfVerts = (Vector3 *)MemAlloc(sizeof(Vector3) * triCount * 3);
    surfCols = (Color *)MemAlloc(sizeof(Color) * triCount * 3);
    surfaceMesh.colors = (unsigned char *)MemAlloc(sizeof(unsigned char) * triCount * 12);
    surfaceMesh.normals = (float *)MemAlloc(sizeof(float) * triCount * 9);
    for (int i = 0; i < triCount * 3; i++) {
        surfaceMesh.normals[i * 3 + 0] = 0.0f;
        surfaceMesh.normals[i * 3 + 1] = 1.0f;
        surfaceMesh.normals[i * 3 + 2] = 0.0f;
    }
    surfaceMesh.texcoords = (float *)MemAlloc(sizeof(float) * triCount * 6);
    for (int i = 0; i < triCount * 6; i++) surfaceMesh.texcoords[i] = 0.0f;
    UploadMesh(&surfaceMesh, false);
    surfaceMeshInit = true;
}

/* depth-based water color (fallback path + bulk particles) */
static Color water_color(float y, float foam) {
    float t = (y - 0.0f) / 5.0f;
    if (t < 0.0f) t = 0.0f;
    if (t > 1.0f) t = 1.0f;
    int r = (int)(10 + 40 * t);
    int g = (int)(70 + 130 * t);
    int b = (int)(140 + 110 * t);
    if (t > 0.7f) { r += (int)(40 * (t - 0.7f) / 0.3f); g += (int)(20 * (t - 0.7f) / 0.3f); }
    if (foam > 1.0f) foam = 1.0f;
    r = (int)(r + (235 - r) * foam);
    g = (int)(g + (245 - g) * foam);
    b = (int)(b + (250 - b) * foam);
    return (Color){ (unsigned char)r, (unsigned char)g, (unsigned char)b, 255 };
}

static float foam_at(World *w, float x, float z) {
    int gx = (int)((x + ARENA) / WAV_CELL);
    int gz = (int)((z + ARENA) / WAV_CELL);
    if (gx < 0 || gz < 0 || gx >= WAV_N || gz >= WAV_N) return 0.0f;
    return w->waveGrid[gx][gz].foam;
}

static float water_h(World *w, float x, float z) {
    return water_height_at(w, x, z);
}

/* analytic AO field: splat solid bodies into grid, done once per frame */
static float aoGrid[WAV_N][WAV_N];

static void build_ao_grid(World *w) {
    static float occ[WAV_N][WAV_N];
    for (int i = 0; i < WAV_N; i++)
        for (int j = 0; j < WAV_N; j++) occ[i][j] = 0.0f;
    for (int b = 0; b < w->bodyCount; b++) {
        Body *bd = &w->bodies[b];
        if (bd->tagged == 3) continue;
        int gx = (int)((bd->pos.x + ARENA) / WAV_CELL);
        int gz = (int)((bd->pos.z + ARENA) / WAV_CELL);
        if (gx < -4 || gz < -4 || gx > WAV_N + 3 || gz > WAV_N + 3) continue;
        float rc = (bd->radius + 0.7f) / WAV_CELL;
        int r = (int)(rc + 0.5f);
        if (r < 1) r = 1;
        if (r > 4) r = 4;
        for (int di = -r; di <= r; di++)
            for (int dj = -r; dj <= r; dj++) {
                int ii = gx + di, jj = gz + dj;
                if (ii < 0 || jj < 0 || ii >= WAV_N || jj >= WAV_N) continue;
                occ[ii][jj] += 1.0f / (1.0f + 0.5f * (float)(di * di + dj * dj));
            }
    }
    for (int i = 0; i < WAV_N; i++)
        for (int j = 0; j < WAV_N; j++) {
            float ao = 1.0f / (1.0f + 0.5f * occ[i][j]);
            if (ao < 0.45f) ao = 0.45f;
            aoGrid[i][j] = ao;
        }
}

static float ao_at(float x, float z) {
    int gx = (int)((x + ARENA) / WAV_CELL);
    int gz = (int)((z + ARENA) / WAV_CELL);
    if (gx < 0 || gz < 0 || gx >= WAV_N || gz >= WAV_N) return 1.0f;
    return aoGrid[gx][gz];
}

/* wave-grid central-difference normal at cell (i,j) */
static void grid_normal(World *w, int i, int j, float *nx, float *ny, float *nz) {
    int i0 = i > 0 ? i - 1 : i, i1 = i < WAV_N - 1 ? i + 1 : i;
    int j0 = j > 0 ? j - 1 : j, j1 = j < WAV_N - 1 ? j + 1 : j;
    float dhx = w->waveGrid[i1][j].h - w->waveGrid[i0][j].h;
    float dhz = w->waveGrid[i][j1].h - w->waveGrid[i][j0].h;
    float sx = dhx / ((float)(i1 - i0) * WAV_CELL);
    float sz = dhz / ((float)(j1 - j0) * WAV_CELL);
    float il = 1.0f / sqrtf(sx * sx + 1.0f + sz * sz);
    *nx = -sx * il; *ny = il; *nz = -sz * il;
}

static void surface_mesh_update(World *w) {
    int v = 0;
    for (int i = 0; i < WAV_N - 1; i++)
        for (int j = 0; j < WAV_N - 1; j++) {
            int wetAny = w->waveGrid[i][j].wet     || w->waveGrid[i+1][j].wet ||
                         w->waveGrid[i][j+1].wet   || w->waveGrid[i+1][j+1].wet;
            float x0 = -ARENA + i * WAV_CELL;
            float z0 = -ARENA + j * WAV_CELL;
            float x1 = x0 + WAV_CELL;
            float z1 = z0 + WAV_CELL;
            if (!wetAny) {
                for (int k = 0; k < 6; k++) {
                    surfVerts[v] = (Vector3){x0, -0.05f, z0};
                    surfaceMesh.vertices[v*3+0] = x0;
                    surfaceMesh.vertices[v*3+1] = -0.05f;
                    surfaceMesh.vertices[v*3+2] = z0;
                    surfaceMesh.normals[v*3+0] = 0.0f;
                    surfaceMesh.normals[v*3+1] = 1.0f;
                    surfaceMesh.normals[v*3+2] = 0.0f;
                    surfaceMesh.texcoords[v*2+0] = 1.0f;
                    surfaceMesh.texcoords[v*2+1] = 0.0f;
                    surfaceMesh.colors[v*4+0] = 0;
                    surfaceMesh.colors[v*4+1] = 0;
                    surfaceMesh.colors[v*4+2] = 0;
                    surfaceMesh.colors[v*4+3] = 0;
                    v++;
                }
                continue;
            }
            float h00 = w->waveGrid[i][j].h;
            float h10 = w->waveGrid[i + 1][j].h;
            float h11 = w->waveGrid[i + 1][j + 1].h;
            float h01 = w->waveGrid[i][j + 1].h;
            float f00 = w->waveGrid[i][j].foam;
            float f10 = w->waveGrid[i + 1][j].foam;
            float f11 = w->waveGrid[i + 1][j + 1].foam;
            float f01 = w->waveGrid[i][j + 1].foam;
            Vector3 t[6] = {
                { x0, h00, z0 }, { x1, h10, z0 }, { x1, h11, z1 },
                { x0, h00, z0 }, { x1, h11, z1 }, { x0, h01, z1 }
            };
            float ff[6] = { f00, f10, f11, f00, f11, f01 };
            float hh[6] = { h00, h10, h11, h00, h11, h01 };
            int gi[6] = { i, i + 1, i + 1, i, i + 1, i };
            int gj[6] = { j, j, j + 1, j, j + 1, j + 1 };
            for (int k = 0; k < 6; k++) {
                float nx, ny, nz;
                grid_normal(w, gi[k], gj[k], &nx, &ny, &nz);
                float foam = ff[k];
                if (foam < 0.0f) foam = 0.0f;
                if (foam > 1.0f) foam = 1.0f;
                foam *= 0.65f;
                float ao = aoGrid[gi[k]][gj[k]];
                surfVerts[v] = t[k];
                surfaceMesh.vertices[v * 3 + 0] = t[k].x;
                surfaceMesh.vertices[v * 3 + 1] = t[k].y;
                surfaceMesh.vertices[v * 3 + 2] = t[k].z;
                surfaceMesh.normals[v * 3 + 0] = nx;
                surfaceMesh.normals[v * 3 + 1] = ny;
                surfaceMesh.normals[v * 3 + 2] = nz;
                surfaceMesh.texcoords[v * 2 + 0] = ao;
                surfaceMesh.texcoords[v * 2 + 1] = foam;
                Color col = water_color(hh[k], ff[k]);
                surfCols[v] = col;
                surfaceMesh.colors[v * 4 + 0] = col.r;
                surfaceMesh.colors[v * 4 + 1] = col.g;
                surfaceMesh.colors[v * 4 + 2] = col.b;
                surfaceMesh.colors[v * 4 + 3] = col.a;
                v++;
            }
        }
    UpdateMeshBuffer(surfaceMesh, 0, surfaceMesh.vertices, surfaceMesh.vertexCount * 3 * sizeof(float), 0);
    UpdateMeshBuffer(surfaceMesh, 1, surfaceMesh.texcoords, surfaceMesh.vertexCount * 2 * sizeof(float), 0);
    UpdateMeshBuffer(surfaceMesh, 2, surfaceMesh.normals, surfaceMesh.vertexCount * 3 * sizeof(float), 0);
    UpdateMeshBuffer(surfaceMesh, 3, surfaceMesh.colors, surfaceMesh.vertexCount * 4, 0);
}

/* display-space sky gradient matching waterFS skyDisp (horizon 190,205,215 /
 * zenith 70,120,190); drawn screen-space so both passes share one code path */
static void draw_sky(void) {
    int h = GetScreenHeight();
    int hor = h / 2;
    DrawRectangleGradientV(0, 0, GetScreenWidth(), hor,
        (Color){ 70, 120, 190, 255 }, (Color){ 190, 205, 215, 255 });
    DrawRectangleGradientV(0, hor, GetScreenWidth(), h - hor,
        (Color){ 190, 205, 215, 255 }, (Color){ 145, 160, 170, 255 });
}

/* caustics on floor: animated interference, killed by foam, shaded by AO */
static float caustic_at(float x, float z, float time) {
    float a = sinf(x * 0.9f + time * 1.3f) * sinf(z * 1.1f - time * 0.9f);
    float b = sinf(x * 1.7f - time * 1.7f + 2.0f) * sinf(z * 0.6f + time * 1.1f);
    float c = a * b;
    if (c < 0.0f) c = 0.0f;
    return c * c;
}

static void draw_caustics(World *w, float time) {
    const float step = 1.2f;
    for (float x = -ARENA; x < ARENA; x += step)
        for (float z = -ARENA; z < ARENA; z += step) {
            float wh = water_h(w, x, z);
            if (wh < 0.0f) continue;
            float cval = caustic_at(x, z, time);
            if (cval < 0.04f) continue;
            float depthFade = 1.0f - fminf(wh / 4.5f, 1.0f);
            float foamK = 1.0f - foam_at(w, x + step * 0.5f, z + step * 0.5f);
            if (foamK < 0.0f) foamK = 0.0f;
            float ao = ao_at(x + step * 0.5f, z + step * 0.5f);
            int bright = (int)(180 * cval * (0.3f + 0.7f * depthFade) * foamK * ao);
            if (bright < 8) continue;
            Color cc = { (unsigned char)(120 + bright * 0.4f),
                         (unsigned char)(190 + bright * 0.3f),
                         (unsigned char)(230), 120 };
            {
                Vector3 p0 = { x, 0.02f, z };
                Vector3 p1 = { x + step, 0.02f, z };
                Vector3 p2 = { x + step, 0.02f, z + step };
                Vector3 p3 = { x, 0.02f, z + step };
                DrawTriangle3D(p0, p1, p2, cc);
                DrawTriangle3D(p0, p2, p3, cc);
            }
        }
}

/* god rays: translucent quads rising from surface, animated */
static void draw_godrays(World *w, float time) {
    for (int i = 0; i < 7; i++) {
        float px = -ARENA * 0.6f + i * 3.4f;
        float sway = sinf(time * 0.7f + i * 1.9f) * 1.5f;
        float wh = water_h(w, px, 0.0f);
        if (wh < 0.5f) continue;
        Color rc = { 190, 220, 255, 22 };
        Vector3 base = { px + sway, wh, 0.0f };
        Vector3 top = { px + sway * 1.6f - 2.5f, wh + 9.0f, 1.0f };
        Vector3 perp = { 0.55f, 0.0f, 0.85f };
        float wid = 0.9f;
        Vector3 b0 = Vector3Add(base, Vector3Scale(perp, -wid));
        Vector3 b1 = Vector3Add(base, Vector3Scale(perp, wid));
        Vector3 t0 = Vector3Add(top, Vector3Scale(perp, -wid * 2.4f));
        Vector3 t1 = Vector3Add(top, Vector3Scale(perp, wid * 2.4f));
        DrawTriangle3D(b0, b1, t1, rc);
        DrawTriangle3D(b0, t1, t0, rc);
        Vector3 b0b = Vector3Add(base, Vector3Scale(perp, -wid * 0.5f));
        Vector3 b1b = Vector3Add(base, Vector3Scale(perp, wid * 0.5f));
        Vector3 top2 = { top.x + 1.0f, top.y, top.z - 2.0f };
        Vector3 t2 = Vector3Add(top2, Vector3Scale(perp, -wid * 1.2f));
        Vector3 t3 = Vector3Add(top2, Vector3Scale(perp, wid * 1.2f));
        DrawTriangle3D(b0b, b1b, t3, rc);
        DrawTriangle3D(b0b, t3, t2, rc);
    }
}

/* bulk water particles: batched camera-facing billboards (1 draw call).
 * NOTE: emitted as raw TRIANGLES (rl QUADS path is unreliable here). */
static void draw_water_particles(World *w, Camera3D cam) {
    (void)cam;
    for (int fi = 0; fi < fg.nFluid; fi++) {
        int i = fg.fluidIdx[fi];
        Body *b = &w->bodies[i];
        float foam = foam_at(w, b->pos.x, b->pos.z) * 0.6f;
        Color col = water_color(b->pos.y, foam);
        col.a = 130;
        DrawSphereEx(b->pos, b->radius * 1.7f, 4, 3, col);
    }
}

static void draw_boxes(World *w) {
    for (int i = 0; i < w->boxCount; i++) {
        StaticBox *bx = &w->boxes[i];
        DrawCubeV(bx->center, (Vector3){bx->half.x * 2, bx->half.y * 2, bx->half.z * 2}, bx->col);
        DrawCubeWiresV(bx->center, (Vector3){bx->half.x * 2, bx->half.y * 2, bx->half.z * 2}, (Color){110, 140, 170, 255});
    }
}

static void draw_constraints(World *w) {
    for (int i = 0; i < w->conCount; i++) {
        Constraint *c = &w->cons[i];
        if (c->hidden) continue;
        Body *A = &w->bodies[c->a], *B = &w->bodies[c->b];
        Color col = c->thick ? (Color){240, 190, 150, 255} : (Color){120, 120, 160, 255};
        DrawLine3D(A->pos, B->pos, col);
        if (c->thick) {
            DrawSphereEx(A->pos, 0.09f, 6, 6, col);
            DrawSphereEx(B->pos, 0.09f, 6, 6, col);
        }
    }
}

static void draw_solids(World *w, int tintSubmerged) {
    for (int i = 0; i < w->bodyCount; i++) {
        Body *b = &w->bodies[i];
        if (b->tagged == 3) continue;
        Color col;
        if (b->invMass == 0.0f)  col = (Color){230, 90, 90, 255};
        else if (b->tagged == 1) col = (Color){250, 210, 180, 255};
        else if (b->tagged == 2) col = (Color){250, 180, 80, 255};
        else                     col = (Color){90, 200, 250, 255};
        if (tintSubmerged) {
            float wh = water_h(w, b->pos.x, b->pos.z);
            if (b->pos.y < wh) col = (Color){ (unsigned char)(col.r * 0.5f + 20),
                                              (unsigned char)(col.g * 0.6f + 40),
                                              (unsigned char)(col.b * 0.7f + 60), 255 };
        }
        DrawSphereEx(b->pos, b->radius, 8, 8, col);
    }
}

/* ---------------- shader water (two-pass) ---------------- */

#define RT_W 1280
#define RT_H 720

static const char *waterVS =
"#version 330\n"
"in vec3 vertexPosition;\n"
"in vec2 vertexTexCoord;\n"
"in vec3 vertexNormal;\n"
"in vec4 vertexColor;\n"
"uniform mat4 mvp;\n"
"uniform mat4 matModel;\n"
"out vec3 wPos;\n"
"out vec3 wNormal;\n"
"out vec2 data;\n"
"void main() {\n"
"    vec4 wp = matModel * vec4(vertexPosition, 1.0);\n"
"    wPos = wp.xyz;\n"
"    wNormal = normalize(vertexNormal);\n"
"    data = vertexTexCoord;\n"
"    gl_Position = mvp * vec4(vertexPosition, 1.0);\n"
"}\n";

static const char *waterFS =
"#version 330\n"
"in vec3 wPos;\n"
"in vec3 wNormal;\n"
"in vec2 data;\n"
"out vec4 finalColor;\n"
"uniform sampler2D sceneTex;\n"
"uniform sampler2D sceneDepth;\n"
"uniform vec3 camPos;\n"
"uniform vec3 sunDir;\n"
"uniform float time;\n"
"uniform vec2 resolution;\n"
"float linDepth(float d) {\n"
"    float n = 0.01;\n"
"    float f = 1000.0;\n"
"    return (2.0 * n * f) / (f + n - (2.0 * d - 1.0) * (f - n));\n"
"}\n"
/* display-space sky: must match the CPU background gradient exactly */
"vec3 skyDisp(vec3 R) {\n"
"    float t = clamp(R.y, 0.0, 1.0);\n"
"    vec3 c = mix(vec3(0.745, 0.804, 0.843), vec3(0.275, 0.471, 0.745), pow(t, 0.55));\n"
"    float sun = pow(max(dot(R, sunDir), 0.0), 60.0);\n"
"    c += vec3(1.0, 0.9, 0.7) * sun * 0.7;\n"
"    return c;\n"
"}\n"
"void main() {\n"
"    vec3 V = camPos - wPos;\n"
"    V = V / max(length(V), 1e-5);\n"
"    vec3 N = normalize(wNormal);\n"
"    float e1 = sin(wPos.x * 3.1 + time * 2.0) * sin(wPos.z * 2.7 - time * 1.6);\n"
"    float e2 = sin(wPos.x * 5.3 - time * 2.6 + 1.7) * sin(wPos.z * 4.1 + time * 2.2);\n"
"    N = normalize(N + vec3(e1 * 0.06 + e2 * 0.03, 0.0, e2 * 0.06 - e1 * 0.03));\n"
"    float ao = clamp(data.x, 0.0, 1.0);\n"
"    float foam = clamp(data.y, 0.0, 1.0);\n"
"    vec2 uv = gl_FragCoord.xy / resolution;\n"
"    float waterD = gl_FragCoord.z;\n"
"    float sceneD = texture(sceneDepth, uv).r;\n"
"    float sceneLin = linDepth(sceneD);\n"
"    float waterLin = linDepth(waterD);\n"
"    if (sceneLin < waterLin - 0.08) discard;\n"
"    float thick = max(sceneLin - waterLin, 0.0);\n"
"    float cosT = max(dot(V, N), 0.0);\n"
"    float fres = 0.02 + 0.98 * pow(1.0 - cosT, 5.0);\n"
"    fres *= (1.0 - foam);\n"
"    vec2 refrUV = clamp(uv + N.xz * (0.045 / (1.0 + thick * 0.5)), vec2(0.001), vec2(0.999));\n"
"    vec3 sceneCol = texture(sceneTex, refrUV).rgb;\n"
"    /* display-space pipeline: background is already display-referred */\n"
"    vec3 transm = exp(-vec3(0.35, 0.10, 0.03) * thick);\n"
"    vec3 refr = sceneCol * transm;\n"
"    vec3 deep = vec3(0.015, 0.13, 0.22);\n"
"    vec3 shal = vec3(0.20, 0.55, 0.60);\n"
"    float bodyMix = clamp(1.0 - exp(-thick * 0.45), 0.0, 1.0) * 0.85;\n"
"    vec3 wtint = mix(shal, deep, clamp(thick * 0.25, 0.0, 1.0));\n"
"    vec3 base = mix(refr, wtint, bodyMix);\n"
"    float sunAmt = max(dot(N, sunDir), 0.0);\n"
"    base += vec3(1.0, 0.9, 0.75) * sunAmt * exp(-thick * 0.6) * 0.20 * (1.0 - foam);\n"
"    vec3 R = reflect(-V, N);\n"
"    vec3 col = mix(base, skyDisp(R), clamp(fres, 0.0, 1.0));\n"
"    vec3 H = normalize(V + sunDir);\n"
"    float spec = pow(max(dot(N, H), 0.0), 280.0) * (1.0 - foam);\n"
"    col += vec3(1.0, 0.95, 0.85) * spec;\n"
"    col = mix(col, vec3(0.90, 0.93, 0.95), foam);\n"
"    col *= ao;\n"
"    float alpha = clamp(thick * 0.55 + fres * 0.85 + foam * 0.9 + 0.25, 0.0, 1.0);\n"
"    alpha *= smoothstep(0.0, 0.12, thick + foam * 0.25 + 0.02);\n"
"    finalColor = vec4(clamp(col, 0.0, 1.0), alpha);\n"
"}\n";

static RenderTexture2D sceneTarget;
static bool targetInit = false;
static unsigned int sceneDepthId = 0;
static Shader waterShader;
static bool useShaders = false;
static int locScene = -1, locDepth = -1, locCam = -1, locSun = -1;
static int locTime = -1, locRes = -1, locMvp = -1, locMatModel = -1;
static Material defaultMat;

static void water_backend_init(void) {
    defaultMat = LoadMaterialDefault();
    sceneTarget = LoadRenderTexture(RT_W, RT_H);
    SetTextureFilter(sceneTarget.texture, TEXTURE_FILTER_BILINEAR);
    sceneDepthId = rlLoadTextureDepth(RT_W, RT_H, false);
    if (sceneTarget.id == 0 || sceneDepthId == 0) { useShaders = false; return; }
    rlFramebufferAttach(sceneTarget.id, sceneDepthId, RL_ATTACHMENT_DEPTH, RL_ATTACHMENT_TEXTURE2D, 0);
    targetInit = true;
    waterShader = LoadShaderFromMemory(waterVS, waterFS);
    if (waterShader.id == 0) { useShaders = false; return; }
    locScene = GetShaderLocation(waterShader, "sceneTex");
    locDepth = GetShaderLocation(waterShader, "sceneDepth");
    locCam = GetShaderLocation(waterShader, "camPos");
    locSun = GetShaderLocation(waterShader, "sunDir");
    locTime = GetShaderLocation(waterShader, "time");
    locRes = GetShaderLocation(waterShader, "resolution");
    locMvp = GetShaderLocation(waterShader, "mvp");
    locMatModel = GetShaderLocation(waterShader, "matModel");
    if (locScene < 0 || locDepth < 0 || locCam < 0 || locSun < 0 ||
        locTime < 0 || locRes < 0 || locMvp < 0 || locMatModel < 0) {
        useShaders = false;
        return;
    }
    useShaders = true;
}

static void water_backend_unload(void) {
    if (waterShader.id != 0) UnloadShader(waterShader);
    if (sceneDepthId != 0) rlUnloadTexture(sceneDepthId);
    if (targetInit) UnloadRenderTexture(sceneTarget);
}

/* manual VAO draw with explicit texture units (no Material slot games) */
static void draw_water_surface(Camera3D cam, float time) {
    Matrix matView = rlGetMatrixModelview();
    Matrix matProj = rlGetMatrixProjection();
    Matrix mvp = MatrixMultiply(matView, matProj);
    float cp[3] = { cam.position.x, cam.position.y, cam.position.z };
    float sun[3] = { -0.3907f, 0.8931f, -0.2233f };
    float res[2] = { (float)RT_W, (float)RT_H };
    SetShaderValueMatrix(waterShader, locMvp, mvp);
    {
        Matrix ident = MatrixIdentity();
        SetShaderValueMatrix(waterShader, locMatModel, ident);
    }
    SetShaderValue(waterShader, locCam, cp, SHADER_UNIFORM_VEC3);
    SetShaderValue(waterShader, locSun, sun, SHADER_UNIFORM_VEC3);
    SetShaderValue(waterShader, locTime, &time, SHADER_UNIFORM_FLOAT);
    SetShaderValue(waterShader, locRes, res, SHADER_UNIFORM_VEC2);
    rlDrawRenderBatchActive();
    rlEnableShader(waterShader.id);
    rlSetUniformMatrix(locMvp, mvp);
    {
        Matrix ident = MatrixIdentity();
        rlSetUniformMatrix(locMatModel, ident);
    }
    rlSetUniform(locCam, cp, SHADER_UNIFORM_VEC3, 1);
    rlSetUniform(locSun, sun, SHADER_UNIFORM_VEC3, 1);
    rlSetUniform(locTime, &time, SHADER_UNIFORM_FLOAT, 1);
    rlSetUniform(locRes, res, SHADER_UNIFORM_VEC2, 1);
    rlActiveTextureSlot(1);
    rlEnableTexture(sceneTarget.texture.id);
    {
        int u = 1;
        rlSetUniform(locScene, &u, SHADER_UNIFORM_INT, 1);
    }
    rlActiveTextureSlot(2);
    rlEnableTexture(sceneDepthId);
    {
        int u = 2;
        rlSetUniform(locDepth, &u, SHADER_UNIFORM_INT, 1);
    }
    rlDisableBackfaceCulling();
    /* screen depth holds the 2D background blit: rely on manual depth
     * discard in the shader instead of the (stale) depth buffer */
    rlDisableDepthTest();
    rlEnableVertexArray(surfaceMesh.vaoId);
    rlDrawVertexArray(0, surfaceMesh.vertexCount);
    rlDisableVertexArray();
    rlActiveTextureSlot(1);
    rlDisableTexture();
    rlActiveTextureSlot(2);
    rlDisableTexture();
    rlActiveTextureSlot(0);
    rlDisableShader();
    rlEnableDepthTest();
    rlEnableBackfaceCulling();
}

int main(void) {
    InitWindow(1280, 720, "Deterministic 3D Verlet Physics - raylib");
    SetTargetFPS(60);

    unsigned long long seed = 42;
    static World world;
    world_reset(&world, seed);
    surface_mesh_init();
    water_backend_init();
    useShaders = useShaders && (getenv("NOSHADER") == NULL);

    Camera3D cam = { 0 };
    cam.target = (Vector3){ 0, 3, 0 };
    cam.up = (Vector3){ 0, 1, 0 };
    cam.fovy = 45.0f;
    cam.projection = CAMERA_PERSPECTIVE;

    float camYaw = 45.0f, camPitch = 0.6f, camDist = 30.0f;
    char status[256];
    Color floorCol = { 140, 125, 95, 255 };

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_R)) { world_reset(&world, seed); grabbed = -1; }
        if (IsKeyPressed(KEY_N)) { seed++; world_reset(&world, seed); grabbed = -1; }

        if (IsMouseButtonDown(MOUSE_RIGHT_BUTTON)) {
            Vector2 dm = GetMouseDelta();
            camYaw += dm.x * 0.4f;
            camPitch += dm.y * 0.4f;
            if (camPitch > 1.4f) camPitch = 1.4f;
            if (camPitch < 0.05f) camPitch = 0.05f;
        }
        float wheel = GetMouseWheelMove();
        if (wheel != 0.0f) {
            camDist -= wheel * 2.0f;
            if (camDist < 5.0f) camDist = 5.0f;
            if (camDist > 90.0f) camDist = 90.0f;
        }
        cam.position = (Vector3){
            cam.target.x + camDist * cosf(camPitch) * cosf(camYaw * DEG2RAD),
            cam.target.y + camDist * sinf(camPitch),
            cam.target.z + camDist * cosf(camPitch) * sinf(camYaw * DEG2RAD)
        };

        Ray mouseRay = GetMouseRay(GetMousePosition(), cam);
        if (IsMouseButtonPressed(MOUSE_LEFT_BUTTON)) {
            float best = 1e9f;
            grabbed = -1;
            for (int i = 0; i < world.bodyCount; i++) {
                Body *b = &world.bodies[i];
                Vector3 toB = Vector3Subtract(b->pos, mouseRay.position);
                float t = Vector3DotProduct(toB, mouseRay.direction);
                if (t < 0.0f) continue;
                Vector3 closest = Vector3Add(mouseRay.position, Vector3Scale(mouseRay.direction, t));
                float d = Vector3Distance(closest, b->pos);
                if (d < best && d < b->radius + 0.8f) { best = d; grabbed = i; }
            }
            if (grabbed >= 0) grabTarget = world.bodies[grabbed].pos;
        }
        if (grabbed >= 0 && fabsf(mouseRay.direction.y) > 1e-6f) {
            float t = (grabTarget.y - mouseRay.position.y) / mouseRay.direction.y;
            if (t < 0.0f) t = 0.0f;
            grabTarget = Vector3Add(mouseRay.position, Vector3Scale(mouseRay.direction, t));
        }
        if (IsMouseButtonReleased(MOUSE_LEFT_BUTTON)) grabbed = -1;

        double frame = GetFrameTime();
        if (frame > MAX_ACCUM) frame = MAX_ACCUM;
        world.accumulator += frame;
        while (world.accumulator >= (double)FIXED_DT) {
            apply_mouse(&world);
            world_step(&world);
            world.accumulator -= (double)FIXED_DT;
        }

        snprintf(status, sizeof(status),
                 "seed: %llu steps: %llu bodies: %d fps: %d %s [R] reset [N] seed LMB drag RMB orbit",
                 seed, world.stepCounter, world.bodyCount, GetFPS(),
                 useShaders ? "[SHADER]" : "[FALLBACK]");

        float time = (float)world.stepCounter * FIXED_DT;
        build_ao_grid(&world);
        surface_mesh_update(&world);

        BeginDrawing();
        if (useShaders) {
            /* PASS 1: scene (no water surface) -> FBO */
            BeginTextureMode(sceneTarget);
            ClearBackground((Color){18, 18, 24, 255});
            draw_sky();
            BeginMode3D(cam);
                DrawPlane((Vector3){ 0.0f, -0.01f, 0.0f }, (Vector2){ 40.0f, 40.0f }, floorCol);
                DrawGrid(40, 1.0f);
                rlDisableBackfaceCulling();
                draw_caustics(&world, time);
                rlEnableBackfaceCulling();
                draw_boxes(&world);
                draw_constraints(&world);
                draw_water_particles(&world, cam);
                draw_solids(&world, 0);
            EndMode3D();
            EndTextureMode();
            /* PASS 2: composite to screen */
            ClearBackground((Color){0, 0, 0, 255});
            draw_sky();
            DrawTextureRec(sceneTarget.texture,
                (Rectangle){ 0, 0, (float)RT_W, -(float)RT_H },
                (Vector2){ 0, 0 }, WHITE);
            BeginMode3D(cam);
                draw_water_surface(cam, time);
                draw_godrays(&world, time);
                if (grabbed >= 0)
                    DrawSphereWires(grabTarget, 0.5f, 6, 6, (Color){255, 220, 90, 255});
            EndMode3D();
        } else {
            ClearBackground((Color){18, 18, 24, 255});
            draw_sky();
            BeginMode3D(cam);
                DrawPlane((Vector3){ 0.0f, -0.01f, 0.0f }, (Vector2){ 40.0f, 40.0f }, floorCol);
                DrawGrid(40, 1.0f);
                rlDisableBackfaceCulling();
                draw_caustics(&world, time);
                draw_boxes(&world);
                draw_constraints(&world);
                draw_water_particles(&world, cam);
                draw_solids(&world, 1);
                rlDisableBackfaceCulling();
                DrawMesh(surfaceMesh, defaultMat, MatrixIdentity());
                rlEnableBackfaceCulling();
                draw_godrays(&world, time);
                rlEnableBackfaceCulling();
                if (grabbed >= 0)
                    DrawSphereWires(grabTarget, 0.5f, 6, 6, (Color){255, 220, 90, 255});
            EndMode3D();
        }
        DrawText(status, 12, 12, 18, (Color){200, 200, 210, 255});
        EndDrawing();

        /* debug: screenshot at intervals (shot0 = startup, later = settled) */
        if (world.stepCounter >= 90 && world.stepCounter <= 200 && world.stepCounter % 10 == 0) {
            char fn[32]; snprintf(fn, sizeof(fn), "seq%03d.png", (int)world.stepCounter);
            TakeScreenshot(fn);
            int wet = 0;
            for (int wi = 0; wi < WAV_N; wi++)
                for (int wj = 0; wj < WAV_N; wj++) wet += world.waveGrid[wi][wj].wet ? 1 : 0;
            Vector3 fp = fg.nFluid > 0 ? world.bodies[fg.fluidIdx[0]].pos : (Vector3){0,0,0};
            fprintf(stderr, "DIAG step=%llu bodies=%d nFluid=%d wet=%d f0=(%.2f,%.2f,%.2f) shaders=%d\n",
                world.stepCounter, world.bodyCount, fg.nFluid, wet, fp.x, fp.y, fp.z, useShaders ? 1 : 0);
        }
    }

    water_backend_unload();
    if (surfaceMeshInit) UnloadMesh(surfaceMesh);
    CloseWindow();
    return 0;
}
