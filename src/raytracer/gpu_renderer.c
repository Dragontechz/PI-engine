#define Material RtTracerMaterial
#include "gpu_renderer.h"
#undef Material
#include "gpu_opencl.h"

#include "raylib.h"
#include "rlgl.h"
#include "raymath.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define GPU_MAX_SPHERES 24
#define GPU_MAX_BOXES 24
#define GPU_MAX_PLANES 8
#define GPU_MAX_LIGHTS RT_MAX_LIGHTS
#define GPU_TRI_TEX_W 1024

#define STRINGIFY_IMPL(x) #x
#define STRINGIFY(x) STRINGIFY_IMPL(x)

struct GpuRenderer {
    int backend; /* 0 = GLSL/OpenGL, 1 = OpenCL */
    void *ocl;
    Shader shader;
    RenderTexture2D target;
    Model quad;
    Texture2D tris_tex;
    int tri_count;
    int tris_uploaded;
    int broken;
    int loc_cam_pos, loc_fwd, loc_right, loc_up, loc_aspect_tan;
    int loc_counts, loc_tri_count;
    int loc_sph, loc_sph_mat, loc_sph_emi;
    int loc_box_min, loc_box_max, loc_box_mat, loc_box_emi;
    int loc_plane_mat, loc_plane_emi;
    int loc_light, loc_light_col;
    int loc_tris, loc_bunny, loc_bunny_mat, loc_bunny_emi;
};

static const char *GPU_VS =
    "#version 330\n"
    "layout(location = 0) in vec3 vertexPosition;\n"
    "layout(location = 1) in vec2 vertexTexCoord;\n"
    "out vec2 fragTexCoord;\n"
    "void main() {\n"
    "    fragTexCoord = vertexTexCoord;\n"
    "    gl_Position = vec4(vertexPosition, 1.0);\n"
    "}\n";

static const char *GPU_FS =
    "#version 330\n"
    "in vec2 fragTexCoord;\n"
    "out vec4 fragColor;\n"
    "\n"
    "uniform vec3 uCamPos;\n"
    "uniform vec3 uFwd;\n"
    "uniform vec3 uRight;\n"
    "uniform vec3 uUp;\n"
    "uniform vec2 uAspectTan;\n"
    "uniform ivec4 uCounts;\n"
    "uniform int uTriCount;\n"
    "uniform vec4 uSph[" STRINGIFY(GPU_MAX_SPHERES) "];\n"
    "uniform vec4 uSphMat[" STRINGIFY(GPU_MAX_SPHERES) "];\n"
    "uniform vec4 uSphEmi[" STRINGIFY(GPU_MAX_SPHERES) "];\n"
    "uniform vec4 uBoxMin[" STRINGIFY(GPU_MAX_BOXES) "];\n"
    "uniform vec4 uBoxMax[" STRINGIFY(GPU_MAX_BOXES) "];\n"
    "uniform vec4 uBoxMat[" STRINGIFY(GPU_MAX_BOXES) "];\n"
    "uniform vec4 uBoxEmi[" STRINGIFY(GPU_MAX_BOXES) "];\n"
    "uniform vec4 uPlaneMat[" STRINGIFY(GPU_MAX_PLANES) "];\n"
    "uniform vec4 uPlaneEmi[" STRINGIFY(GPU_MAX_PLANES) "];\n"
    "uniform vec4 uLight[" STRINGIFY(GPU_MAX_LIGHTS) "];\n"
    "uniform vec4 uLightCol[" STRINGIFY(GPU_MAX_LIGHTS) "];\n"
    "uniform sampler2D uTris;\n"
    "uniform vec4 uBunny;\n"
    "uniform vec4 uBunnyMat;\n"
    "uniform vec4 uBunnyEmi;\n"
    "\n"
    "#define EPS 0.0005f\n"
    "struct Hit { float t; vec3 n; int kind; int idx; };\n"
    "\n"
    "vec2 sph_hit(vec4 s, vec3 ro, vec3 rd) {\n"
    "    vec3 oc = ro - s.xyz;\n"
    "    float b = dot(oc, rd);\n"
    "    float c = dot(oc, oc) - s.w * s.w;\n"
    "    float disc = b * b - c;\n"
    "    if (disc < 0.0) return vec2(-1.0);\n"
    "    float sq = sqrt(disc);\n"
    "    float t0 = -b - sq, t1 = -b + sq;\n"
    "    if (t0 > EPS) return vec2(t0, t1);\n"
    "    if (t1 > EPS) return vec2(t1, t0);\n"
    "    return vec2(-1.0);\n"
    "}\n"
    "\n"
    "vec2 box_hit(vec4 mn, vec4 mx, vec3 ro, vec3 rd) {\n"
    "    vec3 inv = 1.0 / rd;\n"
    "    vec3 t0 = (mn.xyz - ro) * inv;\n"
    "    vec3 t1 = (mx.xyz - ro) * inv;\n"
    "    vec3 tn = min(t0, t1), tf = max(t0, t1);\n"
    "    float near = max(max(tn.x, tn.y), tn.z);\n"
    "    float far = min(min(tf.x, tf.y), tf.z);\n"
    "    if (near > far || far < EPS) return vec2(-1.0);\n"
    "    return vec2(near > EPS ? near : far, near);\n"
    "}\n"
    "\n"
    "float plane_hit(float y, vec3 ro, vec3 rd) {\n"
    "    if (abs(rd.y) < 1e-9) return -1.0;\n"
    "    float t = (y - ro.y) / rd.y;\n"
    "    return t > EPS ? t : -1.0;\n"
    "}\n"
    "\n"
    "vec3 tri_fetch(int tri, int k) {\n"
    "    int texel = tri * 3 + k;\n"
    "    ivec2 uv = ivec2(texel % " STRINGIFY(GPU_TRI_TEX_W) ", texel / " STRINGIFY(GPU_TRI_TEX_W) ");\n"
    "    return texelFetch(uTris, uv, 0).xyz;\n"
    "}\n"
    "\n"
    "float tri_hit(int tri, vec3 ro, vec3 rd) {\n"
    "    vec3 v0 = tri_fetch(tri, 0);\n"
    "    vec3 e1 = tri_fetch(tri, 1);\n"
    "    vec3 e2 = tri_fetch(tri, 2);\n"
    "    vec3 p = cross(rd, e2);\n"
    "    float det = dot(e1, p);\n"
    "    if (abs(det) < 1e-9) return -1.0;\n"
    "    float inv = 1.0 / det;\n"
    "    vec3 tv = ro - v0;\n"
    "    float u = dot(tv, p) * inv;\n"
    "    if (u < 0.0 || u > 1.0) return -1.0;\n"
    "    vec3 q = cross(tv, e1);\n"
    "    float v = dot(rd, q) * inv;\n"
    "    if (v < 0.0 || u + v > 1.0) return -1.0;\n"
    "    float t = dot(e2, q) * inv;\n"
    "    return t > EPS ? t : -1.0;\n"
    "}\n"
    "\n"
    "bool bunny_might_hit(vec3 ro, vec3 rd) {\n"
    "    if (uTriCount == 0) return false;\n"
    "    vec4 s = vec4(uBunny.xyz, uBunny.w * 1.02);\n"
    "    return sph_hit(s, ro, rd).x > 0.0;\n"
    "}\n"
    "\n"
    "Hit trace(vec3 ro, vec3 rd) {\n"
    "    Hit h; h.t = 1e30; h.kind = -1; h.idx = -1; h.n = vec3(0.0, 1.0, 0.0);\n"
    "    for (int i = 0; i < uCounts.x; i++) {\n"
    "        vec2 t = sph_hit(uSph[i], ro, rd);\n"
    "        if (t.x > 0.0 && t.x < h.t) {\n"
    "            vec3 p = ro + rd * t.x;\n"
    "            h.t = t.x; h.n = normalize(p - uSph[i].xyz); h.kind = 0; h.idx = i;\n"
    "        }\n"
    "    }\n"
    "    for (int i = 0; i < uCounts.y; i++) {\n"
    "        vec2 t = box_hit(uBoxMin[i], uBoxMax[i], ro, rd);\n"
    "        if (t.x > 0.0 && t.x < h.t) {\n"
    "            vec3 p = ro + rd * t.x;\n"
    "            vec3 c = 0.5 * (uBoxMin[i].xyz + uBoxMax[i].xyz);\n"
    "            vec3 d = (p - c) / max(uBoxMax[i].xyz - c, vec3(1e-6));\n"
    "            vec3 ad = abs(d);\n"
    "            vec3 n = vec3(0.0);\n"
    "            if (ad.x >= ad.y && ad.x >= ad.z) n = vec3(sign(d.x), 0.0, 0.0);\n"
    "            else if (ad.y >= ad.z) n = vec3(0.0, sign(d.y), 0.0);\n"
    "            else n = vec3(0.0, 0.0, sign(d.z));\n"
    "            h.t = t.x; h.n = n; h.kind = 1; h.idx = i;\n"
    "        }\n"
    "    }\n"
    "    for (int i = 0; i < uCounts.z; i++) {\n"
    "        float t = plane_hit(uPlaneMat[i].w, ro, rd);\n"
    "        if (t > 0.0 && t < h.t) {\n"
    "            h.t = t; h.n = vec3(0.0, ro.y > uPlaneMat[i].w ? 1.0 : -1.0, 0.0);\n"
    "            h.kind = 2; h.idx = i;\n"
    "        }\n"
    "    }\n"
    "    if (bunny_might_hit(ro, rd)) {\n"
    "        for (int i = 0; i < uTriCount; i++) {\n"
    "            float t = tri_hit(i, ro, rd);\n"
    "            if (t > 0.0 && t < h.t) {\n"
    "                h.t = t; h.n = normalize(cross(tri_fetch(i, 1), tri_fetch(i, 2)));\n"
    "                h.kind = 3; h.idx = i;\n"
    "            }\n"
    "        }\n"
    "    }\n"
    "    return h;\n"
    "}\n"
    "\n"
    "bool occluded(vec3 ro, vec3 rd, float max_t) {\n"
    "    for (int i = 0; i < uCounts.x; i++) {\n"
    "        vec2 t = sph_hit(uSph[i], ro, rd);\n"
    "        if (t.x > 0.0 && t.x < max_t) return true;\n"
    "    }\n"
    "    for (int i = 0; i < uCounts.y; i++) {\n"
    "        vec2 t = box_hit(uBoxMin[i], uBoxMax[i], ro, rd);\n"
    "        if (t.x > 0.0 && t.x < max_t) return true;\n"
    "    }\n"
    "    for (int i = 0; i < uCounts.z; i++) {\n"
    "        float t = plane_hit(uPlaneMat[i].w, ro, rd);\n"
    "        if (t > 0.0 && t < max_t) return true;\n"
    "    }\n"
    "    if (bunny_might_hit(ro, rd)) {\n"
    "        for (int i = 0; i < uTriCount; i++) {\n"
    "            float t = tri_hit(i, ro, rd);\n"
    "            if (t > 0.0 && t < max_t) return true;\n"
    "        }\n"
    "    }\n"
    "    return false;\n"
    "}\n"
    "\n"
    "void main() {\n"
    "    float u = fragTexCoord.x * 2.0 - 1.0;\n"
    "    float v = fragTexCoord.y * 2.0 - 1.0;\n"
    "    vec3 rd = normalize(uFwd + uRight * (u * uAspectTan.x) + uUp * (v * uAspectTan.y));\n"
    "    vec3 ro = uCamPos;\n"
    "    Hit h = trace(ro, rd);\n"
    "    vec3 col;\n"
    "    if (h.kind < 0) {\n"
    "        float t = clamp(rd.y * 0.5 + 0.5, 0.0, 1.0);\n"
    "        vec3 horizon = vec3(0.72, 0.78, 0.86);\n"
    "        vec3 zenith = vec3(0.28, 0.42, 0.72);\n"
    "        col = mix(horizon, zenith, t);\n"
    "    } else {\n"
    "        vec3 albedo, emi;\n"
    "        if (h.kind == 0) { albedo = uSphMat[h.idx].rgb; emi = uSphEmi[h.idx].rgb; }\n"
    "        else if (h.kind == 1) { albedo = uBoxMat[h.idx].rgb; emi = uBoxEmi[h.idx].rgb; }\n"
    "        else if (h.kind == 2) { albedo = uPlaneMat[h.idx].rgb; emi = uPlaneEmi[h.idx].rgb; }\n"
    "        else { albedo = uBunnyMat.rgb; emi = uBunnyEmi.rgb; }\n"
    "        vec3 p = ro + rd * h.t;\n"
    "        col = albedo * 0.14 + emi;\n"
    "        for (int li = 0; li < uCounts.w; li++) {\n"
    "            vec3 to_l = uLight[li].xyz - p;\n"
    "            float d2 = dot(to_l, to_l);\n"
    "            if (uLightCol[li].w < 0.5 && d2 > 49.0) continue;\n"
    "            vec3 l = to_l * inversesqrt(d2);\n"
    "            float ndl = dot(h.n, l);\n"
    "            if (ndl <= 0.0) continue;\n"
    "            if (occluded(p + h.n * EPS, l, sqrt(d2) - EPS)) continue;\n"
    "            col += albedo * ndl * uLightCol[li].rgb * (uLight[li].w / (d2 + 1.0));\n"
    "        }\n"
    "    }\n"
    "    col = clamp(col, 0.0, 1.0);\n"
    "    col = pow(col, vec3(1.0 / 2.2));\n"
    "    fragColor = vec4(col, 1.0);\n"
    "}\n";

static int upload_triangles(GpuRenderer *g, const Scene *s) {
    if (g->tris_uploaded) return 1;
    int total = 0;
    for (int i = 0; i < s->count; i++) {
        if (s->objects[i].shape == RT_MESH) {
            total += s->meshes[s->objects[i].geometry.mesh.mesh].triangle_count;
        }
    }
    g->tri_count = total;
    if (total <= 0) { g->tris_uploaded = 1; return 1; }
    int texels = total * 3;
    int height = (texels + GPU_TRI_TEX_W - 1) / GPU_TRI_TEX_W;
    float *data = (float *)calloc((size_t)GPU_TRI_TEX_W * height * 4, sizeof(float));
    if (!data) return 0;
    int tri = 0;
    for (int i = 0; i < s->count; i++) {
        if (s->objects[i].shape != RT_MESH) continue;
        const RtMesh *m = &s->meshes[s->objects[i].geometry.mesh.mesh];
        for (int t = 0; t < m->triangle_count; t++, tri++) {
            V3 v0 = m->vertices[m->triangles[t][0]];
            V3 v1 = m->vertices[m->triangles[t][1]];
            V3 v2 = m->vertices[m->triangles[t][2]];
            float *row = data + (size_t)tri * 12;
            row[0] = v0.x; row[1] = v0.y; row[2] = v0.z;
            row[4] = v1.x - v0.x; row[5] = v1.y - v0.y; row[6] = v1.z - v0.z;
            row[8] = v2.x - v0.x; row[9] = v2.y - v0.y; row[10] = v2.z - v0.z;
        }
    }
    Image img = {
        .data = data, .width = GPU_TRI_TEX_W, .height = height,
        .mipmaps = 1, .format = PIXELFORMAT_UNCOMPRESSED_R32G32B32A32
    };
    g->tris_tex = LoadTextureFromImage(img);
    free(data);
    if (g->tris_tex.id == 0) return 0;
    g->tris_uploaded = 1;
    return 1;
}

static void set_uniform_v(Shader sh, int loc, const float *v, int count) {
    if (loc >= 0) SetShaderValueV(sh, loc, v, SHADER_UNIFORM_VEC4, count);
}

GpuRenderer *GpuRenderer_Create(int width, int height) {
    GpuRenderer *g = (GpuRenderer *)calloc(1, sizeof *g);
    if (!g) return NULL;
    /* Preferred backend: OpenCL (direct GPU compute, no GL interop). */
    g->ocl = Ocl_Create(width, height);
    if (g->ocl) {
        g->backend = 1;
        TraceLog(LOG_INFO, "GPU renderer: OpenCL backend active");
        return g;
    }
    TraceLog(LOG_WARNING, "GPU renderer: OpenCL unavailable, trying GLSL/OpenGL");
    g->backend = 0;
    g->shader = LoadShaderFromMemory(GPU_VS, GPU_FS);
    if (g->shader.id <= 0 || g->shader.locs == NULL) {
        TraceLog(LOG_WARNING, "GPU renderer: shader compile failed, staying on CPU");
        free(g);
        return NULL;
    }
    g->target = LoadRenderTexture(width, height);
    if (g->target.id == 0) {
        UnloadShader(g->shader);
        free(g);
        return NULL;
    }
    Mesh quad = { 0 };
    quad.vertexCount = 4;
    quad.triangleCount = 2;
    /* raylib owns and frees the CPU-side arrays inside UnloadModel, so they
     * must live on the heap, not in static storage. */
    static const float verts_init[12] = { -1, -1, 0,  1, -1, 0,  1, 1, 0,  -1, 1, 0 };
    static const float uvs_init[8] = { 0, 1,  1, 1,  1, 0,  0, 0 };
    static const float normals_init[12] = { 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1 };
    static const unsigned char colors_init[16] = {
        255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255
    };
    static const unsigned short idx_init[6] = { 0, 1, 2, 0, 2, 3 };
    quad.vertices = (float *)malloc(sizeof verts_init);
    quad.texcoords = (float *)malloc(sizeof uvs_init);
    quad.normals = (float *)malloc(sizeof normals_init);
    quad.colors = (unsigned char *)malloc(sizeof colors_init);
    quad.indices = (unsigned short *)malloc(sizeof idx_init);
    if (!quad.vertices || !quad.texcoords || !quad.normals || !quad.colors || !quad.indices) {
        free(quad.vertices); free(quad.texcoords); free(quad.normals);
        free(quad.colors); free(quad.indices);
        UnloadRenderTexture(g->target);
        UnloadShader(g->shader);
        free(g);
        return NULL;
    }
    memcpy(quad.vertices, verts_init, sizeof verts_init);
    memcpy(quad.texcoords, uvs_init, sizeof uvs_init);
    memcpy(quad.normals, normals_init, sizeof normals_init);
    memcpy(quad.colors, colors_init, sizeof colors_init);
    memcpy(quad.indices, idx_init, sizeof idx_init);
    /* LoadModelFromMesh does NOT upload to VRAM: without UploadMesh all
     * vaoId/vboId stay 0 and DrawMesh silently draws nothing. */
    UploadMesh(&quad, false);
    g->quad = LoadModelFromMesh(quad);
    g->quad.materials[0].shader = g->shader;

    g->loc_cam_pos = GetShaderLocation(g->shader, "uCamPos");
    g->loc_fwd = GetShaderLocation(g->shader, "uFwd");
    g->loc_right = GetShaderLocation(g->shader, "uRight");
    g->loc_up = GetShaderLocation(g->shader, "uUp");
    g->loc_aspect_tan = GetShaderLocation(g->shader, "uAspectTan");
    g->loc_counts = GetShaderLocation(g->shader, "uCounts");
    g->loc_tri_count = GetShaderLocation(g->shader, "uTriCount");
    g->loc_sph = GetShaderLocation(g->shader, "uSph");
    g->loc_sph_mat = GetShaderLocation(g->shader, "uSphMat");
    g->loc_sph_emi = GetShaderLocation(g->shader, "uSphEmi");
    g->loc_box_min = GetShaderLocation(g->shader, "uBoxMin");
    g->loc_box_max = GetShaderLocation(g->shader, "uBoxMax");
    g->loc_box_mat = GetShaderLocation(g->shader, "uBoxMat");
    g->loc_box_emi = GetShaderLocation(g->shader, "uBoxEmi");
    g->loc_plane_mat = GetShaderLocation(g->shader, "uPlaneMat");
    g->loc_plane_emi = GetShaderLocation(g->shader, "uPlaneEmi");
    g->loc_light = GetShaderLocation(g->shader, "uLight");
    g->loc_light_col = GetShaderLocation(g->shader, "uLightCol");
    g->loc_tris = GetShaderLocation(g->shader, "uTris");
    g->loc_bunny = GetShaderLocation(g->shader, "uBunny");
    g->loc_bunny_mat = GetShaderLocation(g->shader, "uBunnyMat");
    g->loc_bunny_emi = GetShaderLocation(g->shader, "uBunnyEmi");
    TraceLog(LOG_WARNING, "GPU locs: cam=%d fwd=%d right=%d up=%d aspect=%d counts=%d tricnt=%d sph=%d tris=%d bunny=%d",
             g->loc_cam_pos, g->loc_fwd, g->loc_right, g->loc_up, g->loc_aspect_tan,
             g->loc_counts, g->loc_tri_count, g->loc_sph, g->loc_tris, g->loc_bunny);
    TraceLog(LOG_INFO, "GPU renderer ready");
    return g;
}

int GpuRenderer_Render(GpuRenderer *g, const Scene *s, unsigned char *rgb,
                       int width, int height, int internal_w, int internal_h,
                       int spp, float cas) {
    if (!g || g->broken || !s || !rgb || width <= 0 || height <= 0) return 0;
    if (g->backend == 1) {
        if (Ocl_Render((OclRenderer *)g->ocl, s, rgb, width, height,
                       internal_w, internal_h, spp, cas)) return 1;
        /* OpenCL failed mid-run: drop it and let GLSL/CPU take over. */
        TraceLog(LOG_WARNING, "GPU renderer: OpenCL failed, falling back");
        Ocl_Destroy((OclRenderer *)g->ocl);
        g->ocl = NULL;
        g->backend = 0;
        if (g->shader.id == 0) return 0; /* GLSL backend never initialized */
    }
    static float sph[GPU_MAX_SPHERES * 4], sph_mat[GPU_MAX_SPHERES * 4], sph_emi[GPU_MAX_SPHERES * 4];
    static float box_min[GPU_MAX_BOXES * 4], box_max[GPU_MAX_BOXES * 4];
    static float box_mat[GPU_MAX_BOXES * 4], box_emi[GPU_MAX_BOXES * 4];
    static float plane_mat[GPU_MAX_PLANES * 4], plane_emi[GPU_MAX_PLANES * 4];
    static float light[GPU_MAX_LIGHTS * 4], light_col[GPU_MAX_LIGHTS * 4];

    int ns = 0, nb = 0, np = 0;
    for (int i = 0; i < s->count; i++) {
        const Object *o = &s->objects[i];
        if (o->shape == RT_SPHERE && ns < GPU_MAX_SPHERES) {
            sph[ns * 4 + 0] = o->geometry.sphere.center.x;
            sph[ns * 4 + 1] = o->geometry.sphere.center.y;
            sph[ns * 4 + 2] = o->geometry.sphere.center.z;
            sph[ns * 4 + 3] = o->geometry.sphere.radius;
            sph_mat[ns * 4 + 0] = o->material.albedo.x;
            sph_mat[ns * 4 + 1] = o->material.albedo.y;
            sph_mat[ns * 4 + 2] = o->material.albedo.z;
            sph_emi[ns * 4 + 0] = o->material.emission.x;
            sph_emi[ns * 4 + 1] = o->material.emission.y;
            sph_emi[ns * 4 + 2] = o->material.emission.z;
            ns++;
        } else if (o->shape == RT_BOX && nb < GPU_MAX_BOXES) {
            box_min[nb * 4 + 0] = o->geometry.box.min.x;
            box_min[nb * 4 + 1] = o->geometry.box.min.y;
            box_min[nb * 4 + 2] = o->geometry.box.min.z;
            box_max[nb * 4 + 0] = o->geometry.box.max.x;
            box_max[nb * 4 + 1] = o->geometry.box.max.y;
            box_max[nb * 4 + 2] = o->geometry.box.max.z;
            box_mat[nb * 4 + 0] = o->material.albedo.x;
            box_mat[nb * 4 + 1] = o->material.albedo.y;
            box_mat[nb * 4 + 2] = o->material.albedo.z;
            box_emi[nb * 4 + 0] = o->material.emission.x;
            box_emi[nb * 4 + 1] = o->material.emission.y;
            box_emi[nb * 4 + 2] = o->material.emission.z;
            nb++;
        } else if (o->shape == RT_PLANE && np < GPU_MAX_PLANES) {
            plane_mat[np * 4 + 0] = o->material.albedo.x;
            plane_mat[np * 4 + 1] = o->material.albedo.y;
            plane_mat[np * 4 + 2] = o->material.albedo.z;
            plane_mat[np * 4 + 3] = o->geometry.plane.y;
            plane_emi[np * 4 + 0] = o->material.emission.x;
            plane_emi[np * 4 + 1] = o->material.emission.y;
            plane_emi[np * 4 + 2] = o->material.emission.z;
            np++;
        }
    }
    int nl = s->light_count < GPU_MAX_LIGHTS ? s->light_count : GPU_MAX_LIGHTS;
    for (int i = 0; i < nl; i++) {
        light[i * 4 + 0] = s->lights[i].position.x;
        light[i * 4 + 1] = s->lights[i].position.y;
        light[i * 4 + 2] = s->lights[i].position.z;
        light[i * 4 + 3] = s->lights[i].power;
        light_col[i * 4 + 0] = s->lights[i].color.x;
        light_col[i * 4 + 1] = s->lights[i].color.y;
        light_col[i * 4 + 2] = s->lights[i].color.z;
        light_col[i * 4 + 3] = (float)(s->lights[i].cast_shadows != 0);
    }

    V3 fwd = vnorm(vsub(s->target, s->camera));
    V3 right = vnorm(vcross(fwd, v3(0.0f, 1.0f, 0.0f)));
    V3 up = vcross(right, fwd);
    float aspect = (float)width / (float)height;
    float tan_h = tanf(s->fov * 3.14159265f / 360.0f);
    float aspect_tan[2] = { aspect * tan_h, tan_h };
    float cam_pos[3] = { s->camera.x, s->camera.y, s->camera.z };
    float f3[3] = { fwd.x, fwd.y, fwd.z }, r3[3] = { right.x, right.y, right.z };
    float u3[3] = { up.x, up.y, up.z };
    int counts[4] = { ns, nb, np, nl };

    if (!upload_triangles(g, s)) { g->broken = 1; return 0; }

    BeginTextureMode(g->target);
    ClearBackground(BLACK);
    set_uniform_v(g->shader, g->loc_sph, sph, GPU_MAX_SPHERES);
    set_uniform_v(g->shader, g->loc_sph_mat, sph_mat, GPU_MAX_SPHERES);
    set_uniform_v(g->shader, g->loc_sph_emi, sph_emi, GPU_MAX_SPHERES);
    set_uniform_v(g->shader, g->loc_box_min, box_min, GPU_MAX_BOXES);
    set_uniform_v(g->shader, g->loc_box_max, box_max, GPU_MAX_BOXES);
    set_uniform_v(g->shader, g->loc_box_mat, box_mat, GPU_MAX_BOXES);
    set_uniform_v(g->shader, g->loc_box_emi, box_emi, GPU_MAX_BOXES);
    set_uniform_v(g->shader, g->loc_plane_mat, plane_mat, GPU_MAX_PLANES);
    set_uniform_v(g->shader, g->loc_plane_emi, plane_emi, GPU_MAX_PLANES);
    set_uniform_v(g->shader, g->loc_light, light, nl);
    set_uniform_v(g->shader, g->loc_light_col, light_col, nl);
    SetShaderValue(g->shader, g->loc_cam_pos, cam_pos, SHADER_UNIFORM_VEC3);
    SetShaderValue(g->shader, g->loc_fwd, f3, SHADER_UNIFORM_VEC3);
    SetShaderValue(g->shader, g->loc_right, r3, SHADER_UNIFORM_VEC3);
    SetShaderValue(g->shader, g->loc_up, u3, SHADER_UNIFORM_VEC3);
    SetShaderValue(g->shader, g->loc_aspect_tan, aspect_tan, SHADER_UNIFORM_VEC2);
    SetShaderValue(g->shader, g->loc_counts, counts, SHADER_UNIFORM_IVEC4);
    SetShaderValue(g->shader, g->loc_tri_count, &g->tri_count, SHADER_UNIFORM_INT);
    if (g->tri_count > 0) {
        V3 mn = v3(1e30f, 1e30f, 1e30f), mx = v3(-1e30f, -1e30f, -1e30f);
        const Object *mesh_obj = NULL;
        for (int i = 0; i < s->count; i++) {
            if (s->objects[i].shape != RT_MESH) continue;
            mesh_obj = &s->objects[i];
            const RtMesh *m = &s->meshes[s->objects[i].geometry.mesh.mesh];
            for (int v = 0; v < m->vertex_count; v++) {
                mn.x = fminf(mn.x, m->vertices[v].x); mx.x = fmaxf(mx.x, m->vertices[v].x);
                mn.y = fminf(mn.y, m->vertices[v].y); mx.y = fmaxf(mx.y, m->vertices[v].y);
                mn.z = fminf(mn.z, m->vertices[v].z); mx.z = fmaxf(mx.z, m->vertices[v].z);
            }
        }
        V3 bc = vscale(vadd(mn, mx), 0.5f);
        V3 br = vscale(vsub(mx, mn), 0.5f);
        float bunny[4] = { bc.x, bc.y, bc.z, vlen(br) };
        float bmat[4] = { 0.9f, 0.88f, 0.85f, 0.0f }, bemi[4] = { 0, 0, 0, 0 };
        if (mesh_obj) {
            bmat[0] = mesh_obj->material.albedo.x;
            bmat[1] = mesh_obj->material.albedo.y;
            bmat[2] = mesh_obj->material.albedo.z;
            bemi[0] = mesh_obj->material.emission.x;
            bemi[1] = mesh_obj->material.emission.y;
            bemi[2] = mesh_obj->material.emission.z;
        }
        SetShaderValue(g->shader, g->loc_bunny, bunny, SHADER_UNIFORM_VEC4);
        SetShaderValue(g->shader, g->loc_bunny_mat, bmat, SHADER_UNIFORM_VEC4);
        SetShaderValue(g->shader, g->loc_bunny_emi, bemi, SHADER_UNIFORM_VEC4);
        rlActiveTextureSlot(1);
        rlEnableTexture(g->tris_tex.id);
        int unit = 1;
        SetShaderValue(g->shader, g->loc_tris, &unit, SHADER_UNIFORM_INT);
    }
    DrawMesh(g->quad.meshes[0], g->quad.materials[0], MatrixIdentity());
    EndTextureMode();

    Image img = LoadImageFromTexture(g->target.texture);
    if (!img.data) { g->broken = 1; return 0; }
    static int dump_once = 0;
    if (!dump_once) {
        dump_once = 1;
        const unsigned char *d = (const unsigned char *)img.data;
        TraceLog(LOG_WARNING, "GPU first px: %d %d %d | counts ns=%d nb=%d np=%d nl=%d tri=%d cam=(%.2f %.2f %.2f) fwd=(%.2f %.2f %.2f) tan=(%.2f %.2f)",
                 d[0], d[1], d[2], ns, nb, np, nl, g->tri_count,
                 cam_pos[0], cam_pos[1], cam_pos[2], f3[0], f3[1], f3[2],
                 aspect_tan[0], aspect_tan[1]);
    }
    const unsigned char *pix = (const unsigned char *)img.data;
    for (int y = 0; y < height; y++) {
        const unsigned char *row = pix + (size_t)y * width * 4;
        unsigned char *out = rgb + (size_t)y * width * 3;
        for (int x = 0; x < width; x++) {
            out[x * 3 + 0] = row[x * 4 + 0];
            out[x * 3 + 1] = row[x * 4 + 1];
            out[x * 3 + 2] = row[x * 4 + 2];
        }
    }
    UnloadImage(img);
    return 1;
}

void GpuRenderer_Destroy(GpuRenderer *g) {
    if (!g) return;
    if (g->ocl) {
        Ocl_Destroy((OclRenderer *)g->ocl);
        free(g);
        return;
    }
    UnloadTexture(g->tris_tex);
    UnloadModel(g->quad);
    UnloadRenderTexture(g->target);
    UnloadShader(g->shader);
    free(g);
}
