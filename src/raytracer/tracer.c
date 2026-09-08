/* Ray tracer core: analytic intersections, sampled CPU transport, GGX/Fresnel
 * shading, finite-light visibility, HDR post-processing, and sky.
 * See tracer.h for the API contract. */
#include "tracer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if defined(__GNUC__)
#define RT_UNUSED __attribute__((unused))
#else
#define RT_UNUSED
#endif
#if defined(__AVX2__)
#include <immintrin.h>
#endif

static float ffminf_(float a, float b) { return a < b ? a : b; }
static float ffmaxf_(float a, float b) { return a > b ? a : b; }
#if defined(__AVX2__)
static unsigned char fast_gamma_lut[4096];
static int fast_gamma_ready;

static RT_UNUSED void init_fast_gamma(void) {
    if (fast_gamma_ready) return;
    for (int i = 0; i < 4096; i++) {
        float x = (float)i / 4095.0f;
        float y = sqrtf(sqrtf(x));
        float gamma = y * (0.18f + 0.82f * y);
        fast_gamma_lut[i] = (unsigned char)(gamma * 255.0f + 0.5f);
    }
    fast_gamma_ready = 1;
}
#endif

static float texture_hash(int x, int y, int z) {
    unsigned int n = (unsigned int)x * 374761393u;
    n += (unsigned int)y * 668265263u;
    n += (unsigned int)z * 2147483647u;
    n = (n ^ (n >> 13)) * 1274126177u;
    n ^= n >> 16;
    return (float)(n & 0xffffu) / 65535.0f;
}

#define RT_PI 3.14159265358979323846f
#define RT_LUMINANCE_TO_CD_M2 100.0f
#define RT_EXPOSURE_KEY 0.18f
#define RT_PUPIL_REFERENCE_MM 4.0f

static float rt_frame_delta = 1.0f / 45.0f;
static float rt_adapted_luminance = -1.0f;

static V3 material_albedo(const Object *o, V3 p);
static V3 sky_color(const Scene *s, V3 rd);
static int rt_occluded_direction(const Scene *s, V3 origin, V3 direction, float distance);
static unsigned char linear_to_srgb(float x);
static void render_camera_basis(const Scene *s, int width, int height,
                                V3 *fwd, V3 *right, V3 *up, float *aspect, float *tanH);
static V3 camera_ray(V3 fwd, V3 right, V3 up, float aspect, float tanH,
                     int x, int y, int width, int height);
static int postprocess_hdr(const V3 *hdr, unsigned char *rgb, int width, int height);

void rt_set_frame_delta(float delta_seconds) {
    if (delta_seconds > 0.0f && isfinite(delta_seconds)) {
        rt_frame_delta = delta_seconds < 0.25f ? delta_seconds : 0.25f;
    }
}

static float clamp01(float x) {
    return x < 0.0f ? 0.0f : (x > 1.0f ? 1.0f : x);
}

static float luminance(V3 c) {
    return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
}

static V3 fresnel_schlick(V3 f0, float cos_theta) {
    float m = 1.0f - clamp01(cos_theta);
    float factor = m * m * m * m * m;
    return vadd(f0, vscale(vsub(v3(1, 1, 1), f0), factor));
}

static float ggx_distribution(float no_h, float roughness) {
    float alpha = roughness * roughness;
    float alpha2 = alpha * alpha;
    float no_h2 = no_h * no_h;
    float denominator = no_h2 * (alpha2 - 1.0f) + 1.0f;
    return alpha2 / (RT_PI * denominator * denominator);
}

static float smith_schlick_g1(float no_v, float roughness) {
    float k = (roughness + 1.0f) * (roughness + 1.0f) * 0.125f;
    return no_v / (no_v * (1.0f - k) + k);
}

static float smith_visibility(float no_l, float no_v, float roughness) {
    return smith_schlick_g1(no_l, roughness) * smith_schlick_g1(no_v, roughness);
}

static V3 surface_direct(V3 albedo, V3 normal, V3 view, V3 light_dir,
                         V3 light_color, float light_power, float distance_sq,
                         float shadow_fraction, float reflection, float shininess) {
    float no_l = ffmaxf_(vdot(normal, light_dir), 0.0f);
    float no_v = ffmaxf_(vdot(normal, view), 0.0f);
    if (no_l <= 0.0f || no_v <= 0.0f || shadow_fraction <= 0.0f) return v3(0, 0, 0);

    V3 half_vector = vnorm(vadd(light_dir, view));
    float no_h = ffmaxf_(vdot(normal, half_vector), 0.0f);
    float vo_h = ffmaxf_(vdot(view, half_vector), 0.0f);
    float roughness = sqrtf(2.0f / (shininess + 2.0f));
    if (roughness < 0.045f) roughness = 0.045f;
    if (roughness > 1.0f) roughness = 1.0f;
    V3 f0 = vadd(vscale(v3(0.04f, 0.04f, 0.04f), 1.0f - reflection),
                 vscale(albedo, reflection));
    V3 fresnel = fresnel_schlick(f0, vo_h);
    float d = ggx_distribution(no_h, roughness);
    float g = smith_visibility(no_l, no_v, roughness);
    V3 specular = vscale(fresnel, d * g / ffmaxf_(4.0f * no_l * no_v, 1e-5f));
    V3 diffuse = vscale(vmul(albedo, vsub(v3(1, 1, 1), fresnel)), 1.0f / RT_PI);
    V3 brdf = vadd(diffuse, specular);
    float irradiance = light_power * no_l / (4.0f * RT_PI * ffmaxf_(distance_sq, 1e-4f));
    return vscale(vmul(brdf, light_color), irradiance * shadow_fraction);
}

static unsigned int sobol_owen(unsigned int index, unsigned int dimension) {
    unsigned int value = 0;
    unsigned int direction = 0x80000000u >> (dimension & 7u);
    for (unsigned int bit = 0; index && bit < 32; bit++, index >>= 1) {
        if (index & 1u) value ^= direction;
        direction = (direction >> 1) ^ (0x80200003u & (unsigned int)-(int)(direction & 1u));
    }
    value ^= value * 0x3d20adeau;
    value ^= value >> 11;
    value *= 0x05526c56u;
    value ^= value >> 13;
    return value;
}

static float rng_float(unsigned int *state) {
    unsigned int dimension = (*state) & 7u;
    unsigned int index = (*state) >> 3;
    *state += 1u;
    return ((float)sobol_owen(index + 1u, dimension) + 0.5f) / 4294967296.0f;
}

static unsigned int pixel_seed(int x, int y) {
    unsigned int seed = (unsigned int)(x + 1) * 0x9e3779b9u;
    seed ^= (unsigned int)(y + 1) * 0x85ebca6bu;
    seed ^= 0xc2b2ae35u;
    return ((seed % 65521u) << 3) | 0u;
}

static V3 blackbody_rgb(float kelvin) {
    if (!(kelvin > 0.0f)) return v3(1, 1, 1);
    float t = kelvin / 100.0f;
    float r = t <= 66.0f ? 1.0f : 1.2929362f * powf(t - 60.0f, -0.1332048f);
    float g = t <= 66.0f ? 0.3900816f * logf(t) - 0.6318414f
                         : 1.1298909f * powf(t - 60.0f, -0.0755148f);
    float b = t >= 66.0f ? 1.0f : (t <= 19.0f ? 0.0f : 0.5432068f * logf(t - 10.0f) - 1.1962541f);
    return v3(clamp01(r), clamp01(g), clamp01(b));
}

static V3 light_rgb(const RtPointLight *light) {
    return light->temperature > 0.0f ? blackbody_rgb(light->temperature) : light->color;
}

static void tangent_basis(V3 n, V3 *u, V3 *v) {
    V3 helper = fabsf(n.y) < 0.9f ? v3(0, 1, 0) : v3(1, 0, 0);
    *u = vnorm(vcross(helper, n));
    *v = vcross(n, *u);
}

static V3 sample_cosine(V3 n, unsigned int *rng, float *pdf) {
    float r = sqrtf(rng_float(rng));
    float phi = 2.0f * RT_PI * rng_float(rng);
    float z = sqrtf(ffmaxf_(1.0f - r * r, 0.0f));
    V3 u, v;
    tangent_basis(n, &u, &v);
    V3 d = vadd(vadd(vscale(u, r * cosf(phi)), vscale(v, r * sinf(phi))), vscale(n, z));
    *pdf = z / RT_PI;
    return vnorm(d);
}

static V3 sample_ggx(V3 n, float roughness, unsigned int *rng, float *pdf) {
    float alpha = roughness * roughness;
    float phi = 2.0f * RT_PI * rng_float(rng);
    float u = rng_float(rng);
    float cos_theta = sqrtf((1.0f - u) / (1.0f + (alpha * alpha - 1.0f) * u));
    float sin_theta = sqrtf(ffmaxf_(1.0f - cos_theta * cos_theta, 0.0f));
    V3 tangent, bitangent;
    tangent_basis(n, &tangent, &bitangent);
    V3 h = vnorm(vadd(vadd(vscale(tangent, sin_theta * cosf(phi)),
                            vscale(bitangent, sin_theta * sinf(phi))),
                       vscale(n, cos_theta)));
    *pdf = ggx_distribution(ffmaxf_(vdot(n, h), 0.0f), roughness) /
           ffmaxf_(4.0f * fabsf(vdot(h, n)), 1e-5f);
    return h;
}

static V3 brdf_eval(V3 albedo, V3 n, V3 view, V3 light, float reflection,
                    float shininess, float *pdf) {
    float no_l = ffmaxf_(vdot(n, light), 0.0f);
    float no_v = ffmaxf_(vdot(n, view), 0.0f);
    if (no_l <= 0.0f || no_v <= 0.0f) {
        *pdf = 0.0f;
        return v3(0, 0, 0);
    }
    V3 h = vnorm(vadd(light, view));
    float no_h = ffmaxf_(vdot(n, h), 0.0f);
    float vo_h = ffmaxf_(vdot(view, h), 0.0f);
    float roughness = sqrtf(2.0f / (shininess + 2.0f));
    if (roughness < 0.045f) roughness = 0.045f;
    if (roughness > 1.0f) roughness = 1.0f;
    float spec_probability = clamp01(0.25f + 0.70f * reflection);
    V3 f0 = vadd(vscale(v3(0.04f, 0.04f, 0.04f), 1.0f - reflection),
                 vscale(albedo, reflection));
    V3 f = fresnel_schlick(f0, vo_h);
    float d = ggx_distribution(no_h, roughness);
    float g = smith_visibility(no_l, no_v, roughness);
    V3 specular = vscale(f, d * g / ffmaxf_(4.0f * no_l * no_v, 1e-5f));
    V3 diffuse = vscale(vmul(albedo, vsub(v3(1, 1, 1), f)), 1.0f / RT_PI);
    float diffuse_pdf = no_l / RT_PI;
    float spec_pdf = d * no_h / ffmaxf_(4.0f * vo_h, 1e-5f);
    *pdf = (1.0f - spec_probability) * diffuse_pdf + spec_probability * spec_pdf;
    return vadd(diffuse, specular);
}

static V3 sample_bsdf(V3 albedo, V3 n, V3 view, float reflection, float shininess,
                      unsigned int *rng, V3 *direction, float *pdf) {
    float roughness = sqrtf(2.0f / (shininess + 2.0f));
    if (roughness < 0.045f) roughness = 0.045f;
    if (roughness > 1.0f) roughness = 1.0f;
    float spec_probability = clamp01(0.25f + 0.70f * reflection);
    if (rng_float(rng) < spec_probability) {
        V3 h = sample_ggx(n, roughness, rng, pdf);
        *direction = vnorm(vsub(vscale(h, 2.0f * vdot(view, h)), view));
    } else {
        *direction = sample_cosine(n, rng, pdf);
    }
    if (vdot(n, *direction) <= 0.0f) {
        *pdf = 0.0f;
        return v3(0, 0, 0);
    }
    V3 f = brdf_eval(albedo, n, view, *direction, reflection, shininess, pdf);
    return f;
}

static V3 sample_direct_light(const Scene *s, V3 p, V3 n, V3 view, V3 albedo,
                              float reflection, float shininess, unsigned int *rng) {
    if (s->light_count <= 0) return v3(0, 0, 0);
    V3 result = v3(0, 0, 0);
    for (int light_index = 0; light_index < s->light_count; light_index++) {
        const RtPointLight *light = &s->lights[light_index];
        V3 lp = light->position;
        if (light->radius > 0.0f) {
            V3 axis = vnorm(vsub(light->position, p));
            V3 u, v;
            tangent_basis(axis, &u, &v);
            float r = sqrtf(rng_float(rng)) * light->radius;
            float phi = 2.0f * RT_PI * rng_float(rng);
            lp = vadd(lp, vadd(vscale(u, r * cosf(phi)), vscale(v, r * sinf(phi))));
        }
        V3 delta = vsub(lp, p);
        float distance_sq = vdot(delta, delta);
        if (distance_sq <= 1e-6f) continue;
        float distance = sqrtf(distance_sq);
        V3 l = vscale(delta, 1.0f / distance);
        float no_l = ffmaxf_(vdot(n, l), 0.0f);
        if (no_l <= 0.0f || rt_occluded_direction(s, vadd(p, vscale(n, 1e-3f)), l, distance)) continue;
        float bsdf_pdf;
        V3 f = brdf_eval(albedo, n, view, l, reflection, shininess, &bsdf_pdf);
        if (bsdf_pdf <= 0.0f) continue;
        float light_pdf = 1.0f;
        if (light->radius > 0.0f) {
            float area = RT_PI * light->radius * light->radius;
            float facing = ffmaxf_(vdot(vscale(l, -1.0f), vnorm(vsub(light->position, p))), 0.05f);
            light_pdf = distance_sq / (area * facing);
        }
        float mis = light->radius > 0.0f
            ? (light_pdf * light_pdf) / (light_pdf * light_pdf + bsdf_pdf * bsdf_pdf)
            : 1.0f;
        V3 incoming = vscale(light_rgb(light), light->power / (4.0f * RT_PI * distance_sq));
        result = vadd(result, vscale(vmul(f, incoming), no_l * mis / ffmaxf_(light_pdf, 1e-5f)));
    }
    return result;
}

static V3 path_radiance(const Scene *s, V3 ro, V3 rd, unsigned int *rng,
                        int max_depth, int *primary_object) {
    V3 result = v3(0, 0, 0);
    V3 throughput = v3(1, 1, 1);
    for (int depth = 0; depth <= max_depth; depth++) {
        Hit h;
        if (!rt_trace(s, ro, rd, 1e30f, &h)) {
            V3 environment = sky_color(s, rd);
            if (s->fog_density > 0.0f) {
                float transmittance = expf(-s->fog_density * 100.0f);
                result = vadd(result, vscale(vmul(throughput, s->fog_color), 1.0f - transmittance));
                environment = vscale(environment, transmittance);
            }
            result = vadd(result, vmul(throughput, environment));
            break;
        }
        if (depth == 0 && primary_object) *primary_object = h.object;
        const Object *o = &s->objects[h.object];
        V3 p = vadd(ro, vscale(rd, h.t));
        if (s->fog_density > 0.0f) {
            float transmittance = expf(-s->fog_density * h.t);
            result = vadd(result, vscale(vmul(throughput, s->fog_color), 1.0f - transmittance));
            throughput = vscale(throughput, transmittance);
        }
        V3 n = h.normal;
        V3 albedo = material_albedo(o, p);
        if (o->shape == RT_PLANE && o->geometry.plane.checker && o->material.texture == RT_TEX_NONE &&
            (((int)floorf(p.x) + (int)floorf(p.z)) & 1)) albedo = vscale(albedo, 0.35f);
        result = vadd(result, vmul(throughput, o->material.emission));
        /* The sky is an environment emitter. A small cosine-weighted estimate
         * keeps fully shadowed mesh facets visible without adding fake bounce
         * light to the material model. */
        V3 environment = sky_color(s, n);
        result = vadd(result, vscale(vmul(throughput, vmul(albedo, environment)), 0.08f / RT_PI));
        V3 view = vscale(rd, -1.0f);
        result = vadd(result, vmul(throughput, sample_direct_light(
            s, p, n, view, albedo, o->material.reflection, o->material.shininess, rng)));
        if (depth == max_depth) break;
        float bsdf_pdf;
        V3 next;
        V3 f = sample_bsdf(albedo, n, view, o->material.reflection,
                           o->material.shininess, rng, &next, &bsdf_pdf);
        if (bsdf_pdf <= 1e-6f) break;
        float no_l = ffmaxf_(vdot(n, next), 0.0f);
        if (bsdf_pdf <= 1e-6f || no_l <= 0.0f) break;
        throughput = vscale(vmul(throughput, f), no_l / bsdf_pdf);
        if (depth >= 2) {
            float q = clamp01(ffmaxf_(luminance(throughput), ffmaxf_(throughput.x, ffmaxf_(throughput.y, throughput.z))));
            if (rng_float(rng) > q) break;
            throughput = vscale(throughput, 1.0f / ffmaxf_(q, 1e-3f));
        }
        ro = vadd(p, vscale(n, 1e-3f));
        rd = next;
    }
    return result;
}

static V3 trace_pixel_samples(const Scene *s, V3 fwd, V3 right, V3 up,
                              float aspect, float tanH, int x, int y,
                              int width, int height, int samples, int max_depth,
                              float *mean_luminance, int *primary_object) {
    V3 sum = v3(0, 0, 0);
    float luminance_sum = 0.0f;
    int object = -1;
    if (samples < 1) samples = 1;
    for (int sample = 0; sample < samples; sample++) {
        unsigned int rng = pixel_seed(x, y) + (unsigned int)sample * 0x9e3779b9u;
        float jx = rng_float(&rng);
        float jy = rng_float(&rng);
        float u = ((float)x + jx) / (float)width * 2.0f - 1.0f;
        float v = 1.0f - ((float)y + jy) / (float)height * 2.0f;
        V3 rd = vnorm(vadd(fwd, vadd(vscale(right, u * aspect * tanH), vscale(up, v * tanH))));
        int hit_object = -1;
        V3 c = path_radiance(s, s->camera, rd, &rng, max_depth, &hit_object);
        sum = vadd(sum, c);
        luminance_sum += luminance(c);
        if (sample == 0) object = hit_object;
    }
    float inv = 1.0f / (float)samples;
    if (mean_luminance) *mean_luminance = luminance_sum * inv;
    if (primary_object) *primary_object = object;
    return vscale(sum, inv);
}

/* ---- Morton-order 16x16 tile queue (tile scheduling for worker threads) ---- */
#define RT_TILE 16
typedef struct { unsigned int key; int x, y; } RtTile;

/* Interleave the low 16 bits of x (standard bit scatter, part1by1). */
static unsigned int morton_expand(unsigned int x) {
    x &= 0x0000ffffu;
    x = (x ^ (x << 8)) & 0x00ff00ffu;
    x = (x ^ (x << 4)) & 0x0f0f0f0fu;
    x = (x ^ (x << 2)) & 0x33333333u;
    x = (x ^ (x << 1)) & 0x55555555u;
    return x;
}

static int rt_tile_order_cmp(const void *a, const void *b) {
    const RtTile *ta = (const RtTile *)a, *tb = (const RtTile *)b;
    return ta->key < tb->key ? -1 : (ta->key > tb->key ? 1 : 0);
}

/* Render only the listed tile origins (tx, ty pairs) into the HDR film.
 * Worker threads pop tiles dynamically; pixels are independent (per-pixel
 * seeds), so the output is identical to a full rt_render_native sweep. */
int rt_render_tiles_hdr(const Scene *s, V3 *hdr, int width, int height,
                        int spp, int max_depth, const int *tiles_xy, int tile_count) {
    if (!s || !hdr || !tiles_xy || tile_count <= 0 || width <= 0 || height <= 0) return 0;
    V3 fwd, right, up;
    float aspect, tanH;
    render_camera_basis(s, width, height, &fwd, &right, &up, &aspect, &tanH);
    if (spp < 1) spp = 1;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
    for (int t = 0; t < tile_count; t++) {
        const int tx = tiles_xy[t * 2 + 0];
        const int ty = tiles_xy[t * 2 + 1];
        const int x0 = tx * RT_TILE;
        const int y0 = ty * RT_TILE;
        const int x1 = x0 + RT_TILE < width ? x0 + RT_TILE : width;
        const int y1 = y0 + RT_TILE < height ? y0 + RT_TILE : height;
        for (int y = y0; y < y1; y++) {
            for (int x = x0; x < x1; x++) {
                hdr[y * width + x] = trace_pixel_samples(
                    s, fwd, right, up, aspect, tanH, x, y, width, height,
                    spp, max_depth, NULL, NULL);
            }
        }
    }
    return 1;
}

/* Build a Morton-ordered list of tile origins ((tx, ty) pairs, tile x tile
 * pixels each) covering a width x height film. If out_xy is NULL, returns
 * the number of tiles; otherwise writes at most cap tiles and returns the
 * count written, or -1 if the buffer is too small. */
int rt_build_tile_origins(int width, int height, int tile, int *out_xy, int cap) {
    if (tile <= 0 || width <= 0 || height <= 0) return -1;
    int tile_w = (width + tile - 1) / tile;
    int tile_h = (height + tile - 1) / tile;
    int tile_count = tile_w * tile_h;
    if (!out_xy) return tile_count;
    if (tile_count > cap) return -1;
    RtTile *order = (RtTile *)malloc((size_t)tile_count * sizeof *order);
    if (!order) return -1;
    int n = 0;
    for (int ty = 0; ty < tile_h; ty++) {
        for (int tx = 0; tx < tile_w; tx++) {
            order[n].key = morton_expand((unsigned)tx) |
                           (morton_expand((unsigned)ty) << 1);
            order[n].x = tx;
            order[n].y = ty;
            n++;
        }
    }
    qsort(order, (size_t)tile_count, sizeof *order, rt_tile_order_cmp);
    for (int i = 0; i < tile_count; i++) {
        out_xy[i * 2 + 0] = order[i].x;
        out_xy[i * 2 + 1] = order[i].y;
    }
    free(order);
    return tile_count;
}

static int render_native_hdr(const Scene *s, V3 *hdr, int width, int height,
                             int spp, int max_depth) {
    /* Workers pop 16x16 tiles from a Morton-ordered queue: adjacent tiles are
     * spatially close, and dynamic scheduling load-balances for free. */
    int tile_count = rt_build_tile_origins(width, height, RT_TILE, NULL, 0);
    int *tiles_xy = tile_count > 0
        ? (int *)malloc((size_t)tile_count * 2 * sizeof *tiles_xy) : NULL;
    int ok = 0;
    if (tiles_xy &&
        rt_build_tile_origins(width, height, RT_TILE, tiles_xy, tile_count) == tile_count) {
        ok = rt_render_tiles_hdr(s, hdr, width, height, spp, max_depth,
                                 tiles_xy, tile_count);
    }
    free(tiles_xy);
    return ok;
}

static int render_adaptive_hdr(const Scene *s, V3 *hdr, int width, int height,
                               int spp_min, int spp_max, float epsilon,
                               float contrast, int max_depth) {
    const int tile_size = 16;
    int tile_w = (width + tile_size - 1) / tile_size;
    int tile_h = (height + tile_size - 1) / tile_size;
    int tile_count = tile_w * tile_h;
    float *means = (float *)calloc((size_t)width * height, sizeof *means);
    float *m2 = (float *)calloc((size_t)width * height, sizeof *m2);
    int *counts = (int *)calloc((size_t)width * height, sizeof *counts);
    int *objects = (int *)malloc((size_t)width * height * sizeof *objects);
    unsigned char *active = (unsigned char *)calloc((size_t)tile_count, 1);
    if (!means || !m2 || !counts || !objects || !active) {
        free(means); free(m2); free(counts); free(objects); free(active);
        return 0;
    }
    V3 fwd, right, up;
    float aspect, tanH;
    render_camera_basis(s, width, height, &fwd, &right, &up, &aspect, &tanH);
    if (spp_min < 1) spp_min = 1;
    if (spp_max < spp_min) spp_max = spp_min;
    for (int i = 0; i < tile_count; i++) active[i] = 1;

    int sample_count = 0;
    while (sample_count < spp_max) {
        int batch = sample_count == 0 ? 1 : sample_count;
        if (sample_count + batch > spp_max) batch = spp_max - sample_count;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int tile = 0; tile < tile_count; tile++) {
            if (!active[tile]) continue;
            int tx = tile % tile_w;
            int ty = tile / tile_w;
            int x0 = tx * tile_size;
            int y0 = ty * tile_size;
            int x1 = x0 + tile_size < width ? x0 + tile_size : width;
            int y1 = y0 + tile_size < height ? y0 + tile_size : height;
            for (int y = y0; y < y1; y++) {
                for (int x = x0; x < x1; x++) {
                    int pixel = y * width + x;
                    for (int j = 0; j < batch; j++) {
                        unsigned int rng = pixel_seed(x, y) +
                                           (unsigned int)(sample_count + j) * 0x9e3779b9u;
                        float jx = rng_float(&rng);
                        float jy = rng_float(&rng);
                        float u = ((float)x + jx) / (float)width * 2.0f - 1.0f;
                        float v = 1.0f - ((float)y + jy) / (float)height * 2.0f;
                        V3 rd = vnorm(vadd(fwd, vadd(vscale(right, u * aspect * tanH),
                                                   vscale(up, v * tanH))));
                        int object = -1;
                        V3 c = path_radiance(s, s->camera, rd, &rng, max_depth, &object);
                        int n = ++counts[pixel];
                        float value = luminance(c);
                        float delta = value - means[pixel];
                        means[pixel] += delta / (float)n;
                        m2[pixel] += delta * (value - means[pixel]);
                        hdr[pixel] = vadd(hdr[pixel], vscale(vsub(c, hdr[pixel]), 1.0f / (float)n));
                        objects[pixel] = object;
                    }
                }
            }
        }
        sample_count += batch;
        if (sample_count < spp_min) {
            for (int tile = 0; tile < tile_count; tile++) active[tile] = 1;
            continue;
        }
        for (int tile = 0; tile < tile_count; tile++) {
            int tx = tile % tile_w;
            int ty = tile / tile_w;
            int x0 = tx * tile_size;
            int y0 = ty * tile_size;
            int x1 = x0 + tile_size < width ? x0 + tile_size : width;
            int y1 = y0 + tile_size < height ? y0 + tile_size : height;
            float minimum = 1e30f, maximum = -1e30f, worst_se = 0.0f;
            int geometry_jump = 0;
            for (int y = y0; y < y1; y++) {
                for (int x = x0; x < x1; x++) {
                    int pixel = y * width + x;
                    float value = means[pixel];
                    if (value < minimum) minimum = value;
                    if (value > maximum) maximum = value;
                    if (sample_count > 1) {
                        int count = counts[pixel];
                        float se = count > 1
                            ? sqrtf(ffmaxf_(m2[pixel] / (float)(count - 1), 0.0f)) /
                              (sqrtf((float)count) * (value + 1e-3f))
                            : 1e30f;
                        if (se > worst_se) worst_se = se;
                    }
                    if (objects[pixel] < -1) geometry_jump = 1;
                }
            }
            float tile_contrast = (maximum - minimum) / (maximum + minimum + 1e-3f);
            active[tile] = (unsigned char)(sample_count < spp_max &&
                (worst_se > epsilon || tile_contrast > contrast || geometry_jump));
        }
        int any_active = 0;
        for (int tile = 0; tile < tile_count; tile++) any_active |= active[tile] != 0;
        if (!any_active) break;
    }
    free(means); free(m2); free(counts); free(objects); free(active);
    return 1;
}

int rt_render_native(const Scene *s, unsigned char *rgb, int width, int height,
                     int spp, int max_depth) {
    if (!s || !rgb || width <= 0 || height <= 0) return 0;
    V3 *hdr = (V3 *)calloc((size_t)width * height, sizeof *hdr);
    if (!hdr) return 0;
    int ok = render_native_hdr(s, hdr, width, height, spp, max_depth) &&
             postprocess_hdr(hdr, rgb, width, height);
    free(hdr);
    return ok;
}

int rt_postprocess_hdr(const V3 *hdr, unsigned char *rgb, int width, int height) {
    return postprocess_hdr((V3 *)hdr, rgb, width, height);
}

V3 rt_light_rgb(const RtPointLight *light) {
    return light_rgb(light);
}

int rt_render_adaptive(const Scene *s, unsigned char *rgb, int width, int height,
                       int spp_min, int spp_max, float epsilon, float contrast,
                       int max_depth) {
    if (!s || !rgb || width <= 0 || height <= 0) return 0;
    V3 *hdr = (V3 *)calloc((size_t)width * height, sizeof *hdr);
    if (!hdr) return 0;
    int ok = render_adaptive_hdr(s, hdr, width, height, spp_min, spp_max,
                                 epsilon, contrast, max_depth) &&
             postprocess_hdr(hdr, rgb, width, height);
    free(hdr);
    return ok;
}

static V3 material_albedo(const Object *o, V3 p) {
    Material m = o->material;
    if (m.texture == RT_TEX_NONE) return m.albedo;
    float scale = m.texture_scale > 0.0f ? m.texture_scale : 1.0f;
    int ix = (int)floorf(p.x * scale);
    int iy = (int)floorf(p.y * scale);
    int iz = (int)floorf(p.z * scale);
    float value = texture_hash(ix, iy, iz);
    V3 alternate = m.texture_color;
    if (m.texture == RT_TEX_CHECKER) {
        value = ((ix + iz) & 1) ? 1.0f : 0.0f;
        alternate = m.texture_color;
    } else if (m.texture == RT_TEX_GRASS) {
        value = .72f + .28f * value;
    } else if (m.texture == RT_TEX_DIRT) {
        value = .68f + .32f * value;
    } else if (m.texture == RT_TEX_STONE) {
        value = .70f + .30f * value;
    } else if (m.texture == RT_TEX_WALL) {
        value = .82f + .18f * value;
    } else if (m.texture == RT_TEX_ROOF) {
        value = (((ix + iz * 3) & 3) == 0) ? .48f : (.78f + .22f * value);
    }
    value = 1.0f - m.texture_strength + m.texture_strength * value;
    return vadd(vscale(m.albedo, value), vscale(alternate, 1.0f - value));
}

typedef struct { V3 min, max; } Bounds;
static int bounds_hit(Bounds b, V3 ro, V3 rd, float tmax);

static Bounds bounds_empty(void) {
    Bounds b = { v3(1e30f, 1e30f, 1e30f), v3(-1e30f, -1e30f, -1e30f) };
    return b;
}

static Bounds bounds_union(Bounds a, Bounds b) {
    Bounds r = a;
    r.min.x = ffminf_(r.min.x, b.min.x); r.min.y = ffminf_(r.min.y, b.min.y); r.min.z = ffminf_(r.min.z, b.min.z);
    r.max.x = ffmaxf_(r.max.x, b.max.x); r.max.y = ffmaxf_(r.max.y, b.max.y); r.max.z = ffmaxf_(r.max.z, b.max.z);
    return r;
}

static int object_bounds(const Object *o, Bounds *b) {
    if (!o || !b || o->shape == RT_PLANE) return 0;
    switch (o->shape) {
    case RT_BOX:
        b->min = o->geometry.box.min; b->max = o->geometry.box.max; return 1;
    case RT_SPHERE: {
        V3 c = o->geometry.sphere.center; float r = o->geometry.sphere.radius;
        b->min = vsub(c, v3(r, r, r)); b->max = vadd(c, v3(r, r, r)); return 1;
    }
    case RT_CYLINDER: {
        V3 c = o->geometry.cylinder.base; float r = o->geometry.cylinder.radius;
        b->min = v3(c.x-r, c.y, c.z-r); b->max = v3(c.x+r, c.y+o->geometry.cylinder.height, c.z+r); return 1;
    }
    case RT_MESH:
        b->min = o->geometry.mesh.min; b->max = o->geometry.mesh.max; return 1;
    case RT_PLANE: break;
    }
    return 0;
}

static int hit_triangle(V3 ro, V3 rd, V3 a, V3 b, V3 c, float tmax, Hit *hit) {
    V3 e1 = vsub(b, a), e2 = vsub(c, a);
    V3 p = vcross(rd, e2);
    float det = vdot(e1, p);
    if (fabsf(det) < 1e-8f) return 0;
    float inv = 1.0f / det;
    V3 tvec = vsub(ro, a);
    float u = vdot(tvec, p) * inv;
    if (u < 0.0f || u > 1.0f) return 0;
    V3 q = vcross(tvec, e1);
    float v = vdot(rd, q) * inv;
    if (v < 0.0f || u + v > 1.0f) return 0;
    float t = vdot(e2, q) * inv;
    if (t < RT_EPSILON || t > tmax) return 0;
    hit->t = t;
    hit->normal = vnorm(vcross(e1, e2));
    if (vdot(hit->normal, rd) > 0.0f) hit->normal = vscale(hit->normal, -1.0f);
    return 1;
}

static int hit_mesh(const RtMesh *mesh, V3 ro, V3 rd, float tmax, Hit *hit) {
    if (!mesh || mesh->triangle_count < 1 || mesh->triangle_count > RT_MAX_MESH_TRIANGLES) return 0;
    int found = 0;
    float best = tmax;
    if (mesh->node_count <= 0) {
        /* No BVH built: brute-force scan (same closest-hit semantics). */
        for (int i = 0; i < mesh->triangle_count; i++) {
            const int *tri = mesh->triangles[i];
            if (tri[0] < 0 || tri[1] < 0 || tri[2] < 0 ||
                tri[0] >= mesh->vertex_count || tri[1] >= mesh->vertex_count || tri[2] >= mesh->vertex_count) continue;
            Hit candidate;
            if (hit_triangle(ro, rd, mesh->vertices[tri[0]], mesh->vertices[tri[1]], mesh->vertices[tri[2]], best, &candidate)) {
                best = candidate.t;
                *hit = candidate;
                found = 1;
            }
        }
        return found;
    }
    int stack[64]; int sp = 0;
    if (mesh->node_count > 0) stack[sp++] = 0;
    while (sp > 0) {
        const RtMeshNode *node = &mesh->nodes[stack[--sp]];
        Bounds bounds = {node->min, node->max};
        if (!bounds_hit(bounds, ro, rd, best)) continue;
        int start = node->start, end = start + node->count;
        if (node->count == 0) {
            if (node->left >= 0 && sp < 63) stack[sp++] = node->left;
            if (node->right >= 0 && sp < 63) stack[sp++] = node->right;
            continue;
        }
        for (int i = start; i < end; i++) {
        const int *tri = mesh->triangles[mesh->triangle_indices[i]];
        if (tri[0] < 0 || tri[1] < 0 || tri[2] < 0 ||
            tri[0] >= mesh->vertex_count || tri[1] >= mesh->vertex_count || tri[2] >= mesh->vertex_count) continue;
        Hit candidate;
        if (hit_triangle(ro, rd, mesh->vertices[tri[0]], mesh->vertices[tri[1]], mesh->vertices[tri[2]], best, &candidate)) {
            best = candidate.t;
            *hit = candidate;
            found = 1;
        }
        }
    }
    return found;
}

static int bounds_hit(Bounds b, V3 ro, V3 rd, float tmax) {
    float lo = 0.0f, hi = tmax;
    float o[3] = {ro.x, ro.y, ro.z}, d[3] = {rd.x, rd.y, rd.z};
    float mn[3] = {b.min.x, b.min.y, b.min.z}, mx[3] = {b.max.x, b.max.y, b.max.z};
    for (int a = 0; a < 3; a++) {
        if (fabsf(d[a]) < 1e-9f) {
            if (o[a] < mn[a] || o[a] > mx[a]) return 0;
            continue;
        }
        float t0 = (mn[a] - o[a]) / d[a];
        float t1 = (mx[a] - o[a]) / d[a];
        if (t0 > t1) { float t = t0; t0 = t1; t1 = t; }
        if (t0 > lo) lo = t0;
        if (t1 < hi) hi = t1;
        if (lo > hi) return 0;
    }
    return hi >= 0.0f && lo <= tmax;
}

static float bounds_area(Bounds b) {
    float ex = ffmaxf_(b.max.x - b.min.x, 0.0f);
    float ey = ffmaxf_(b.max.y - b.min.y, 0.0f);
    float ez = ffmaxf_(b.max.z - b.min.z, 0.0f);
    return 2.0f * (ex * ey + ey * ez + ez * ex);
}

/* Binned SAH node: bins object centroids along the widest centroid axis and
 * splits at the minimum surface-area-heuristic cost, falling back to a leaf
 * when no split pays for its own traversal. */
static int build_accel_node(Scene *s, int start, int count, int depth) {
    int node_id = s->accel_node_count++;
    RtAccelNode *node = &s->accel_nodes[node_id];
    Bounds all = bounds_empty(), centroid_bounds = bounds_empty();
    for (int i = start; i < start + count; i++) {
        Bounds b;
        if (!object_bounds(&s->objects[s->accel_indices[i]], &b)) continue;
        all = bounds_union(all, b);
        Bounds cb = { vscale(vadd(b.min, b.max), 0.5f), vscale(vadd(b.min, b.max), 0.5f) };
        centroid_bounds = bounds_union(centroid_bounds, cb);
    }
    node->min = all.min; node->max = all.max;
    node->left = node->right = -1; node->start = start; node->count = count;
    if (count <= 2) return node_id;
    if (depth >= RT_SAH_MAX_DEPTH ||
        s->accel_node_count + 2 > (int)(sizeof s->accel_nodes / sizeof s->accel_nodes[0]))
        return node_id;

    float ex = centroid_bounds.max.x - centroid_bounds.min.x;
    float ey = centroid_bounds.max.y - centroid_bounds.min.y;
    float ez = centroid_bounds.max.z - centroid_bounds.min.z;
    int axis = ex > ey && ex > ez ? 0 : (ey > ez ? 1 : 2);
    float cmin = axis == 0 ? centroid_bounds.min.x
               : axis == 1 ? centroid_bounds.min.y : centroid_bounds.min.z;
    float extent = axis == 0 ? ex : (axis == 1 ? ey : ez);
    float node_area = bounds_area(all);
    if (extent <= 1e-6f || node_area <= 0.0f) return node_id;

    Bounds bin_bounds[RT_SAH_BINS];
    int bin_count[RT_SAH_BINS];
    for (int b = 0; b < RT_SAH_BINS; b++) { bin_bounds[b] = bounds_empty(); bin_count[b] = 0; }
    for (int i = start; i < start + count; i++) {
        Bounds b;
        if (!object_bounds(&s->objects[s->accel_indices[i]], &b)) continue;
        V3 c = vscale(vadd(b.min, b.max), 0.5f);
        float t = axis == 0 ? c.x : (axis == 1 ? c.y : c.z);
        int bin = (int)(((t - cmin) / extent) * (float)RT_SAH_BINS);
        if (bin < 0) bin = 0;
        if (bin >= RT_SAH_BINS) bin = RT_SAH_BINS - 1;
        bin_bounds[bin] = bounds_union(bin_bounds[bin], b);
        bin_count[bin]++;
    }

    /* Suffix sweep (bins b..BINS-1), then prefix sweep picking the cheapest split. */
    Bounds suf_bounds[RT_SAH_BINS];
    int suf_count[RT_SAH_BINS];
    Bounds acc_b = bounds_empty();
    int acc_c = 0;
    for (int b = RT_SAH_BINS - 1; b >= 0; b--) {
        acc_b = bounds_union(acc_b, bin_bounds[b]);
        acc_c += bin_count[b];
        suf_bounds[b] = acc_b;
        suf_count[b] = acc_c;
    }
    acc_b = bounds_empty();
    acc_c = 0;
    float best_cost = RT_SAH_PRIM_COST * (float)count;
    int best_split = -1;
    for (int b = 0; b < RT_SAH_BINS - 1; b++) {
        acc_b = bounds_union(acc_b, bin_bounds[b]);
        acc_c += bin_count[b];
        if (acc_c == 0) continue;
        if (suf_count[b + 1] == 0) break;
        float cost = RT_SAH_TRAV_COST + RT_SAH_PRIM_COST *
            ((float)acc_c * bounds_area(acc_b) +
             (float)suf_count[b + 1] * bounds_area(suf_bounds[b + 1])) / node_area;
        if (cost < best_cost) { best_cost = cost; best_split = b; }
    }
    if (best_split < 0) return node_id;

    /* In-place partition: centroids in bins <= best_split go left. */
    int i = start, j = start + count - 1;
    while (i <= j) {
        int id = s->accel_indices[i];
        Bounds b;
        if (!object_bounds(&s->objects[id], &b)) { i++; continue; }
        V3 c = vscale(vadd(b.min, b.max), 0.5f);
        float t = axis == 0 ? c.x : (axis == 1 ? c.y : c.z);
        int bin = (int)(((t - cmin) / extent) * (float)RT_SAH_BINS);
        if (bin < 0) bin = 0;
        if (bin >= RT_SAH_BINS) bin = RT_SAH_BINS - 1;
        if (bin <= best_split) {
            i++;
        } else {
            s->accel_indices[i] = s->accel_indices[j];
            s->accel_indices[j] = id;
            j--;
        }
    }
    int left_count = i - start;
    if (left_count == 0 || left_count == count) return node_id;
    node->left = build_accel_node(s, start, left_count, depth + 1);
    node->right = build_accel_node(s, start + left_count, count - left_count, depth + 1);
    node->count = 0;
    return node_id;
}

void rt_build_accel(Scene *s) {
    if (!s) return;
    s->accel_count = 0;
    s->accel_node_count = 0;
    for (int i = 0; i < s->count; i++) {
        Bounds b;
        if (object_bounds(&s->objects[i], &b)) s->accel_indices[s->accel_count++] = i;
    }
    if (s->accel_count > 0) build_accel_node(s, 0, s->accel_count, 0);
}

/* ---- box (slab method; rays starting inside return the exit hit) ---- */
static int hit_box_obj(const Object *o, V3 ro, V3 rd, float tmax, Hit *hit) {
    V3 mn = o->geometry.box.min, mx = o->geometry.box.max;
    float roa[3] = { ro.x, ro.y, ro.z };
    float rda[3] = { rd.x, rd.y, rd.z };
    float mna[3] = { mn.x, mn.y, mn.z };
    float mxa[3] = { mx.x, mx.y, mx.z };
    float tmin = -1e30f, texit = 1e30f;
    int axMin = 0, axExit = 0, sgnMin = 0, sgnExit = 0;
    for (int a = 0; a < 3; a++) {
        if (fabsf(rda[a]) < 1e-9f) {
            if (roa[a] < mna[a] || roa[a] > mxa[a]) return 0;
            continue;
        }
        float inv = 1.0f / rda[a];
        float t0 = (mna[a] - roa[a]) * inv;
        float t1 = (mxa[a] - roa[a]) * inv;
        int nNear = rda[a] > 0.0f ? -1 : 1;
        if (t0 > t1) { float tt = t0; t0 = t1; t1 = tt; }
        if (t0 > tmin) { tmin = t0; axMin = a; sgnMin = nNear; }
        if (t1 < texit) { texit = t1; axExit = a; sgnExit = rda[a] > 0.0f ? 1 : -1; }
        if (tmin > texit) return 0;
    }
    if (tmin >= RT_EPSILON && tmin <= tmax) {
        hit->t = tmin;
        hit->normal = v3(0, 0, 0);
        (&hit->normal.x)[axMin] = (float)sgnMin;
        return 1;
    }
    if (tmin < RT_EPSILON && texit > RT_EPSILON && texit <= tmax) {
        hit->t = texit;
        hit->normal = v3(0, 0, 0);
        (&hit->normal.x)[axExit] = (float)sgnExit;
        return 1;
    }
    return 0;
}

/* ---- vertical closed cylinder (side + both caps) ---- */
static int hit_cyl_obj(const Object *o, V3 ro, V3 rd, float tmax, Hit *hit) {
    V3 base = o->geometry.cylinder.base;
    float r = o->geometry.cylinder.radius;
    float h = o->geometry.cylinder.height;
    float top = base.y + h;
    float best = 1e30f;
    V3 nBest = v3(0, 0, 0);
    int found = 0;

    float ox = ro.x - base.x, oz = ro.z - base.z;
    float a = rd.x * rd.x + rd.z * rd.z;
    if (a > 1e-12f) {
        float b = 2.0f * (ox * rd.x + oz * rd.z);
        float c = ox * ox + oz * oz - r * r;
        float disc = b * b - 4.0f * a * c;
        if (disc >= 0.0f) {
            float sq = sqrtf(disc);
            for (int s = 0; s < 2; s++) {
                float t = s == 0 ? (-b - sq) / (2.0f * a) : (-b + sq) / (2.0f * a);
                if (t < RT_EPSILON || t > tmax || t >= best) continue;
                float y = ro.y + t * rd.y;
                if (y < base.y || y > top) continue;
                float px = ro.x + t * rd.x - base.x;
                float pz = ro.z + t * rd.z - base.z;
                nBest = vnorm(v3(px, 0, pz));
                best = t;
                found = 1;
            }
        }
    }
    if (fabsf(rd.y) > 1e-9f) {
        for (int cap = 0; cap < 2; cap++) {
            float cy = cap == 0 ? base.y : top;
            float t = (cy - ro.y) / rd.y;
            if (t < RT_EPSILON || t > tmax || t >= best) continue;
            float px = ro.x + t * rd.x - base.x;
            float pz = ro.z + t * rd.z - base.z;
            if (px * px + pz * pz > r * r) continue;
            best = t;
            nBest = cap == 0 ? v3(0, -1, 0) : v3(0, 1, 0);
            found = 1;
        }
    }
    if (!found) return 0;
    hit->t = best;
    hit->normal = nBest;
    return 1;
}

static int hit_sphere_obj(const Object *o, V3 ro, V3 rd, float tmax, Hit *hit) {
    V3 oc = vsub(ro, o->geometry.sphere.center);
    float r = o->geometry.sphere.radius;
    float b = vdot(oc, rd);
    float c = vdot(oc, oc) - r * r;
    float disc = b * b - c;
    if (disc < 0.0f) return 0;
    float sq = sqrtf(disc);
    float t = -b - sq;
    if (t < RT_EPSILON) t = -b + sq; /* inside: exit hit */
    if (t < RT_EPSILON || t > tmax) return 0;
    hit->t = t;
    hit->normal = vnorm(vsub(vadd(ro, vscale(rd, t)), o->geometry.sphere.center));
    return 1;
}

static int hit_plane_obj(const Object *o, V3 ro, V3 rd, float tmax, Hit *hit) {
    float y = o->geometry.plane.y;
    if (fabsf(rd.y) < 1e-9f) return 0;
    float t = (y - ro.y) / rd.y;
    if (t < RT_EPSILON || t > tmax) return 0;
    hit->t = t;
    hit->normal = rd.y > 0.0f ? v3(0, -1, 0) : v3(0, 1, 0);
    return 1;
}

int rt_intersect(const Object *o, V3 ro, V3 rd, float tmax, Hit *hit) {
    if (!o || !hit) return 0;
    switch (o->shape) {
    case RT_BOX: return hit_box_obj(o, ro, rd, tmax, hit);
    case RT_SPHERE: return hit_sphere_obj(o, ro, rd, tmax, hit);
    case RT_CYLINDER: return hit_cyl_obj(o, ro, rd, tmax, hit);
    case RT_PLANE: return hit_plane_obj(o, ro, rd, tmax, hit);
    case RT_MESH: return 0;
    }
    return 0;
}

static int rt_trace_mode(const Scene *s, V3 ro, V3 rd, float tmax, Hit *hit, int render_meshes) {
    if (!s) return 0;
    float best = tmax;
    int idx = -1;
    Hit tmp, best_hit;
    if (s->accel_node_count > 0) {
        int stack[RT_MAX_OBJECTS * 2];
        int sp = 0;
        stack[sp++] = 0;
        while (sp > 0) {
            int node_id = stack[--sp];
            const RtAccelNode *node = &s->accel_nodes[node_id];
            Bounds bounds = { node->min, node->max };
            if (!bounds_hit(bounds, ro, rd, best)) continue;
            if (node->count > 0) {
                for (int i = node->start; i < node->start + node->count; i++) {
                    int object_id = s->accel_indices[i];
                    if (!render_meshes && s->objects[object_id].shape == RT_MESH) continue;
                    int found = s->objects[object_id].shape == RT_MESH
                        ? hit_mesh(&s->meshes[s->objects[object_id].geometry.mesh.mesh], ro, rd, best, &tmp)
                        : rt_intersect(&s->objects[object_id], ro, rd, best, &tmp);
                    if (found) {
                        best = tmp.t;
                        idx = object_id;
                        best_hit = tmp;
                    }
                }
            } else {
                if (node->left >= 0) stack[sp++] = node->left;
                if (node->right >= 0) stack[sp++] = node->right;
            }
        }
        /* Infinite planes are kept outside the finite BVH. */
        for (int i = 0; i < s->count; i++) {
            if (!render_meshes && s->objects[i].shape == RT_MESH) continue;
            if (s->objects[i].shape != RT_PLANE) continue;
            int found = s->objects[i].shape == RT_MESH
                ? hit_mesh(&s->meshes[s->objects[i].geometry.mesh.mesh], ro, rd, best, &tmp)
                : rt_intersect(&s->objects[i], ro, rd, best, &tmp);
            if (found) {
                best = tmp.t;
                idx = i;
                best_hit = tmp;
            }
        }
    } else {
        for (int i = 0; i < s->count; i++) {
            if (!render_meshes && s->objects[i].shape == RT_MESH) continue;
            int found = s->objects[i].shape == RT_MESH
                ? hit_mesh(&s->meshes[s->objects[i].geometry.mesh.mesh], ro, rd, best, &tmp)
                : rt_intersect(&s->objects[i], ro, rd, best, &tmp);
            if (found) {
                best = tmp.t;
                idx = i;
                best_hit = tmp;
            }
        }
    }
    if (idx < 0) return 0;
    *hit = best_hit;
    hit->object = idx;
    return 1;
}

int rt_trace(const Scene *s, V3 ro, V3 rd, float tmax, Hit *hit) {
    return rt_trace_mode(s, ro, rd, tmax, hit, s ? s->render_meshes : 0);
}

int rt_occluded(const Scene *s, V3 p, V3 light) {
    V3 d = vsub(light, p);
    float dist = vlen(d);
    if (dist < 1e-6f) return 0;
    d = vscale(d, 1.0f / dist);
    Hit h;
    return rt_trace(s, p, d, dist - 1e-3f, &h);
}

static int rt_occluded_direction(const Scene *s, V3 origin, V3 direction, float distance) {
    Hit h;
    return rt_trace(s, origin, direction, distance - 1e-3f, &h);
}

static V3 sky_color(const Scene *s, V3 rd) {
    float t = ffmaxf_(ffminf_(rd.y * 0.5f + 0.5f, 1.0f), 0.0f);
    V3 horizon = v3(0.75f, 0.82f, 0.90f);
    V3 zenith = v3(0.25f, 0.45f, 0.78f);
    V3 c = vadd(vscale(horizon, 1.0f - t), vscale(zenith, t));
    V3 ld = vnorm(vsub(s->light, s->camera));
    float glow = ffmaxf_(vdot(rd, ld), 0.0f);
    /* Eight squarings approximate glow^200 without a transcendental call. */
    float glow2 = glow * glow;
    glow2 *= glow2;
    glow2 *= glow2;
    glow2 *= glow2;
    glow2 *= glow2;
    glow2 *= glow2;
    glow2 *= glow2;
    /* A compact sun disk keeps the sky finite while preserving a very bright
     * small source for glare and direct-light sampling. */
    float sun_disk = powf(ffmaxf_(vdot(rd, ld), 0.0f), 9000.0f);
    c = vadd(c, vscale(light_rgb(&(RtPointLight){
        s->light, s->light_color, s->light_power, s->light_radius, 1, 0.0f
    }), glow2 * 0.6f + sun_disk * 120.0f));
    return c;
}

/* Fixed samples across the light disk -> deterministic soft shadows. */
static float shadow_frac_light(const Scene *s, const RtPointLight *light,
                               V3 p, int sample_count) {
    V3 toL = vsub(light->position, p);
    float dist = vlen(toL);
    V3 L = vscale(toL, 1.0f / dist);
    V3 poff = vadd(p, vscale(L, 1e-3f));
    if (sample_count <= 1 || light->radius <= 0.0f) {
        return rt_occluded_direction(s, poff, L, dist) ? 0.0f : 1.0f;
    }
    V3 u = vnorm(vcross(fabsf(L.y) < 0.99f ? v3(0, 1, 0) : v3(1, 0, 0), L));
    V3 v = vcross(L, u);
    float r = light->radius;
    V3 offs[5] = {
        v3(0, 0, 0),
        vscale(u, 0.7f * r), vscale(u, -0.7f * r),
        vscale(v, 0.7f * r), vscale(v, -0.7f * r)
    };
    if (sample_count > 5) sample_count = 5;
    int lit = 0;
    for (int i = 0; i < sample_count; i++) {
        if (!rt_occluded(s, poff, vadd(light->position, offs[i]))) lit++;
    }
    return (float)lit / (float)sample_count;
}

static V3 radiance_ex(const Scene *s, V3 ro, V3 rd, int depth,
                      int shadow_samples, int max_depth, int *primary_object) {
    Hit h;
    if (!rt_trace(s, ro, rd, 1e30f, &h)) {
        if (primary_object) *primary_object = -1;
        return sky_color(s, rd);
    }
    if (primary_object) *primary_object = h.object;
    const Object *o = &s->objects[h.object];
    V3 p = vadd(ro, vscale(rd, h.t));
    /* All analytic intersection routines already return unit normals. */
    V3 n = h.normal;
    V3 albedo = material_albedo(o, p);
    if (o->shape == RT_PLANE && o->geometry.plane.checker && o->material.texture == RT_TEX_NONE) {
        float ck = floorf(p.x) + floorf(p.z);
        if (((int)ck & 1) != 0) albedo = vscale(albedo, 0.35f);
    }

    V3 col = o->material.emission;

    int light_count = s->light_count > 0 ? s->light_count : 1;
    for (int li = 0; li < light_count; li++) {
        RtPointLight fallback = { s->light, s->light_color, s->light_power, s->light_radius, 1, 0.0f };
        const RtPointLight *light = s->light_count > 0 ? &s->lights[li] : &fallback;
        V3 light_delta = vsub(light->position, p);
        float light_distance_sq = vdot(light_delta, light_delta);
        /* Colored portable lights are local emitters. Avoid square roots and
           BSDF work for surfaces outside their useful influence radius. */
        if (!light->cast_shadows && light_distance_sq > 7.0f * 7.0f) continue;
        float frac = light->cast_shadows ? shadow_frac_light(s, light, p, shadow_samples) : 1.0f;
        if (frac <= 0.0f) continue;
        V3 toL = light_delta;
        float dist = sqrtf(light_distance_sq);
        V3 L = vscale(toL, 1.0f / dist);
        V3 view = vscale(rd, -1.0f);
        col = vadd(col, surface_direct(albedo, n, view, L, light->color,
                                       light->power, light_distance_sq, frac,
                                       o->material.reflection, o->material.shininess));
    }

    if (o->material.reflection > 0.0f && depth < max_depth) {
        V3 rdir = vnorm(vsub(rd, vscale(n, 2.0f * vdot(rd, n))));
        V3 rc = radiance_ex(s, vadd(p, vscale(rdir, 1e-3f)), rdir,
                            depth + 1, shadow_samples, max_depth, NULL);
        col = vadd(vscale(col, 1.0f - o->material.reflection),
                   vscale(rc, o->material.reflection));
    }
    return col;
}

/* Exact-pixel interactive fast path for the compact scene. It deliberately
   uses the scene's object list directly: the list is small, immutable between
   ball pushes, and this avoids a per-ray BVH stack plus bounds conversions. */
static RT_UNUSED V3 small_radiance(const Scene *s, V3 ro, V3 rd) {
    float best = 1e30f;
    int object_id = -1;
    Hit candidate;
    for (int i = 0; i < s->count; i++) {
        if (rt_intersect(&s->objects[i], ro, rd, best, &candidate)) {
            best = candidate.t;
            object_id = i;
        }
    }
    if (object_id < 0) return sky_color(s, rd);
    const Object *o = &s->objects[object_id];
    V3 p = vadd(ro, vscale(rd, best));
    V3 n = candidate.normal;
    V3 albedo = material_albedo(o, p);
    if (o->shape == RT_PLANE && o->geometry.plane.checker && o->material.texture == RT_TEX_NONE) {
        if ((((int)floorf(p.x) + (int)floorf(p.z)) & 1) != 0) albedo = vscale(albedo, .35f);
    }
    V3 col = o->material.emission;
    for (int li = 0; li < s->light_count; li++) {
        const RtPointLight *light = &s->lights[li];
        V3 to_l = vsub(light->position, p);
        float d2 = vdot(to_l, to_l);
        if (!light->cast_shadows && d2 > 49.0f) continue;
        float inv_d = 1.0f / sqrtf(d2);
        V3 l = vscale(to_l, inv_d);
        if (light->cast_shadows && rt_occluded_direction(s, vadd(p, vscale(l, .001f)), l, 1.0f/inv_d)) continue;
        col = vadd(col, surface_direct(albedo, n, vscale(rd, -1.0f), l,
                                       light->color, light->power, d2, 1.0f,
                                       o->material.reflection, o->material.shininess));
    }
    if (o->material.reflection > 0.0f) {
        V3 r = vnorm(vsub(rd, vscale(n, 2.0f * vdot(rd, n))));
        Hit rh;
        V3 reflected = sky_color(s, r);
        if (rt_trace(s, vadd(p, vscale(r, .001f)), r, 1e30f, &rh)) {
            reflected = vscale(s->objects[rh.object].material.albedo, .25f);
        }
        col = vadd(vscale(col, 1.0f - o->material.reflection),
                   vscale(reflected, o->material.reflection));
    }
    return col;
}

#if defined(__AVX2__)
static RT_UNUSED V3 small_shade_packet_hit(const Scene *s, V3 ro, V3 rd, float best,
                                 V3 n, int object_id) {
    if (object_id < 0) return sky_color(s, rd);
    const Object *o = &s->objects[object_id];
    V3 p = vadd(ro, vscale(rd, best));
    V3 albedo = material_albedo(o, p);
    if (o->shape == RT_PLANE && o->geometry.plane.checker && o->material.texture == RT_TEX_NONE &&
        ((((int)floorf(p.x) + (int)floorf(p.z)) & 1) != 0))
        albedo = vscale(albedo, .35f);
    V3 col = o->material.emission;
    for (int li = 0; li < s->light_count; li++) {
        const RtPointLight *light = &s->lights[li];
        V3 to_l = vsub(light->position, p);
        float d2 = vdot(to_l, to_l);
        if (!light->cast_shadows && d2 > 49.0f) continue;
        float inv_d = 1.0f / sqrtf(d2);
        V3 l = vscale(to_l, inv_d);
        float ndl = ffmaxf_(vdot(n, l), 0.0f);
        if (ndl <= 0.0f) continue;
        V3 direct = vscale(vmul(albedo, light->color),
                           light->power * ndl / (4.0f * RT_PI * ffmaxf_(d2, 1e-4f)));
        col = vadd(col, direct);
    }
    return col;
}
#endif

V3 rt_radiance(const Scene *s, V3 ro, V3 rd, int depth) {
    return radiance_ex(s, ro, rd, depth, 5, 2, NULL);
}

static unsigned char toByte(float x) {
    if (!(x > -1.0f) || !(x < 10.0f)) x = 0.0f; /* NaN/nonfinite guard */
    int v = (int)(powf(ffmaxf_(ffminf_(x, 1.0f), 0.0f), 1.0f / 2.2f) * 255.0f + 0.5f);
    return (unsigned char)v;
}

static unsigned char linear_to_srgb(float x) {
    if (!(x > 0.0f)) return 0;
    x = ffmaxf_(x, 0.0f);
    float encoded = x <= 0.0031308f
        ? 12.92f * x
        : 1.055f * powf(x, 1.0f / 2.4f) - 0.055f;
    if (encoded > 1.0f) encoded = 1.0f;
    return (unsigned char)(encoded * 255.0f + 0.5f);
}

static V3 exposed_hdr(const V3 *hdr, int width, int height, int x, int y,
                      float exposure) {
    V3 bloom = v3(0, 0, 0);
    static const int offsets[] = { 1, 2, 4, 8 };
    static const float weights[] = { 0.20f, 0.10f, 0.055f, 0.025f };
    for (int i = 0; i < 4; i++) {
        int distance = offsets[i];
        int samples = 0;
        int px[4] = { x - distance, x + distance, x, x };
        int py[4] = { y, y, y - distance, y + distance };
        for (int k = 0; k < 4; k++) {
            if (px[k] < 0 || px[k] >= width || py[k] < 0 || py[k] >= height) continue;
            V3 sample = vscale(hdr[py[k] * width + px[k]], exposure);
            if (luminance(sample) <= 1.0f) continue;
            bloom = vadd(bloom, vscale(sample, weights[i]));
            samples++;
        }
        if (samples > 0) bloom = vscale(bloom, 1.0f / (float)samples);
    }
    return vadd(vscale(hdr[y * width + x], exposure), bloom);
}

/* Shared exposure math: log-avg luminance -> pupil -> adaptation -> exposure.
 * Both the CPU post chain and the GPU post kernels feed this so the temporal
 * adaptation state (rt_adapted_luminance) stays identical whichever backend
 * rendered the frame. log_sum must be the SUM of log(delta + L) over pixels. */
float rt_exposure_from_logavg(double log_sum, int pixel_count) {
    const float delta = 1e-4f;
    if (pixel_count <= 0) return 1.0f;
    float average_luminance = expf((float)(log_sum / (double)pixel_count));
    float cd_m2 = ffmaxf_(average_luminance * RT_LUMINANCE_TO_CD_M2, delta);
    float pupil_mm = 4.9f - 3.0f * tanhf(0.4f * (log10f(cd_m2) - 0.5f));
    if (pupil_mm < 2.0f) pupil_mm = 2.0f;
    if (pupil_mm > 8.0f) pupil_mm = 8.0f;
    float pupil_throughput = (pupil_mm / RT_PUPIL_REFERENCE_MM) *
                             (pupil_mm / RT_PUPIL_REFERENCE_MM);
    float target_luminance = ffmaxf_(average_luminance, delta);
    if (rt_adapted_luminance < 0.0f) rt_adapted_luminance = target_luminance;
    float tau = target_luminance < rt_adapted_luminance ? 4.0f : 0.2f;
    float adaptation = 1.0f - expf(-rt_frame_delta / tau);
    rt_adapted_luminance += adaptation * (target_luminance - rt_adapted_luminance);
    return pupil_throughput * RT_EXPOSURE_KEY / ffmaxf_(rt_adapted_luminance, delta);
}

static int postprocess_hdr(const V3 *hdr, unsigned char *rgb, int width, int height) {
    if (!hdr || !rgb || width <= 0 || height <= 0) return 0;
    const float delta = 1e-4f;
    double log_sum = 0.0;
    int pixel_count = width * height;
    for (int i = 0; i < pixel_count; i++) {
        float l = ffmaxf_(luminance(hdr[i]), 0.0f);
        log_sum += logf(delta + l);
    }
    float exposure = rt_exposure_from_logavg(log_sum, pixel_count);

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            V3 c = exposed_hdr(hdr, width, height, x, y, exposure);
            c = v3(c.x / (1.0f + c.x), c.y / (1.0f + c.y), c.z / (1.0f + c.z));
            int i = (y * width + x) * 3;
            rgb[i + 0] = linear_to_srgb(c.x);
            rgb[i + 1] = linear_to_srgb(c.y);
            rgb[i + 2] = linear_to_srgb(c.z);
        }
    }
    return 1;
}

static unsigned char toByteFast(float x) {
    if (!(x > -1.0f) || !(x < 10.0f)) x = 0.0f;
    x = ffmaxf_(ffminf_(x, 1.0f), 0.0f);
    /* sqrt/sqrt plus a quadratic correction approximates x^(1/2.2). */
    float y = sqrtf(sqrtf(x));
    float gamma = y * (0.18f + 0.82f * y);
    return (unsigned char)(gamma * 255.0f + 0.5f);
}

static void render_camera_basis(const Scene *s, int width, int height,
                                V3 *fwd, V3 *right, V3 *up, float *aspect, float *tanH);

#if defined(__AVX2__)
typedef struct {
    V3 points[3];
    float sx[3], sy[3], sz[3];
    float minx, maxx, miny, maxy;
    float area;
    V3 normal;
    Material material;
} RtScreenTriangle;

static RT_UNUSED void raster_meshes(const Scene *s, unsigned char *rgb, int width, int height) {
    if (!s || s->mesh_count == 0) return;
    float *depth = (float *)malloc((size_t)width * height * sizeof *depth);
    if (!depth) return;
    for (int i = 0; i < width * height; i++) depth[i] = 1e30f;
    V3 fwd, right, up; float aspect, tanH;
    render_camera_basis(s, width, height, &fwd, &right, &up, &aspect, &tanH);
    int total_triangles = 0;
    for (int object_id = 0; object_id < s->count; object_id++) {
        if (s->objects[object_id].shape == RT_MESH) {
            total_triangles += s->meshes[s->objects[object_id].geometry.mesh.mesh].triangle_count;
        }
    }
    RtScreenTriangle *screen = (RtScreenTriangle *)malloc(
        (size_t)total_triangles * sizeof *screen);
    if (!screen) { free(depth); return; }
    int screen_count = 0;
    for (int object_id = 0; object_id < s->count; object_id++) {
        const Object *object = &s->objects[object_id];
        if (object->shape != RT_MESH) continue;
        const RtMesh *mesh = &s->meshes[object->geometry.mesh.mesh];
        for (int ti = 0; ti < mesh->triangle_count; ti++) {
            if (screen_count >= total_triangles) break;
            const int *tr = mesh->triangles[ti];
            RtScreenTriangle *st = &screen[screen_count];
            int visible = 1;
            for (int k = 0; k < 3; k++) {
                st->points[k] = mesh->vertices[tr[k]];
                V3 q = vsub(st->points[k], s->camera);
                st->sz[k] = vdot(q, fwd);
                if (st->sz[k] <= .02f) { visible = 0; break; }
                float u = vdot(q, right) / (st->sz[k] * tanH * aspect);
                float v = vdot(q, up) / (st->sz[k] * tanH);
                st->sx[k] = (u * .5f + .5f) * width;
                st->sy[k] = (1.0f - (v * .5f + .5f)) * height;
            }
            if (!visible) continue;
            st->minx = fmaxf(0.0f, floorf(fminf(st->sx[0], fminf(st->sx[1], st->sx[2]))));
            st->maxx = fminf((float)(width - 1), ceilf(fmaxf(st->sx[0], fmaxf(st->sx[1], st->sx[2]))));
            st->miny = fmaxf(0.0f, floorf(fminf(st->sy[0], fminf(st->sy[1], st->sy[2]))));
            st->maxy = fminf((float)(height - 1), ceilf(fmaxf(st->sy[0], fmaxf(st->sy[1], st->sy[2]))));
            st->area = (st->sx[1] - st->sx[0]) * (st->sy[2] - st->sy[0]) -
                       (st->sy[1] - st->sy[0]) * (st->sx[2] - st->sx[0]);
            if (fabsf(st->area) < 1e-5f) continue;
            st->normal = vnorm(vcross(vsub(st->points[1], st->points[0]),
                                      vsub(st->points[2], st->points[0])));
            st->material = object->material;
            screen_count++;
        }
    }
    int *row_counts = (int *)calloc((size_t)height, sizeof *row_counts);
    int *row_offsets = (int *)malloc((size_t)(height + 1) * sizeof *row_offsets);
    int *row_cursor = (int *)malloc((size_t)height * sizeof *row_cursor);
    if (!row_counts || !row_offsets || !row_cursor) {
        free(row_counts); free(row_offsets); free(row_cursor); free(screen); free(depth);
        return;
    }
    for (int ti = 0; ti < screen_count; ti++) {
        int y0 = (int)screen[ti].miny;
        int y1 = (int)screen[ti].maxy;
        for (int y = y0; y <= y1; y++) row_counts[y]++;
    }
    row_offsets[0] = 0;
    for (int y = 0; y < height; y++) row_offsets[y + 1] = row_offsets[y] + row_counts[y];
    int total_refs = row_offsets[height];
    int *row_triangles = (int *)malloc((size_t)total_refs * sizeof *row_triangles);
    if (!row_triangles) {
        free(row_counts); free(row_offsets); free(row_cursor); free(screen); free(depth);
        return;
    }
    memcpy(row_cursor, row_offsets, (size_t)height * sizeof *row_cursor);
    for (int ti = 0; ti < screen_count; ti++) {
        int y0 = (int)screen[ti].miny;
        int y1 = (int)screen[ti].maxy;
        for (int y = y0; y <= y1; y++) row_triangles[row_cursor[y]++] = ti;
    }
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int y = 0; y < height; y++) {
        for (int ri = row_offsets[y]; ri < row_offsets[y + 1]; ri++) {
            int ti = row_triangles[ri];
            RtScreenTriangle *st = &screen[ti];
            int x0 = (int)st->minx;
            int x1 = (int)st->maxx;
            for (int x = x0; x <= x1; x++) {
                float w0 = ((st->sx[1] - st->sx[0]) * (y - st->sy[0]) -
                            (st->sy[1] - st->sy[0]) * (x - st->sx[0])) / st->area;
                float w1 = ((st->sx[2] - st->sx[1]) * (y - st->sy[1]) -
                            (st->sy[2] - st->sy[1]) * (x - st->sx[1])) / st->area;
                float w2 = 1.0f - w0 - w1;
                if (w0 < 0.0f || w1 < 0.0f || w2 < 0.0f) continue;
                int idx = y * width + x;
                float z = w0 * st->sz[2] + w1 * st->sz[0] + w2 * st->sz[1];
                if (z >= depth[idx]) continue;
                depth[idx] = z;
                V3 point = vadd(vadd(vscale(st->points[2], w0), vscale(st->points[0], w1)),
                                vscale(st->points[1], w2));
                V3 c = vadd(vscale(st->material.albedo, .14f), st->material.emission);
                for (int li = 0; li < s->light_count; li++) {
                    const RtPointLight *light = &s->lights[li];
                    V3 to_l = vsub(light->position, point);
                    float d2 = vdot(to_l, to_l);
                    if (!light->cast_shadows && d2 > 49.0f) continue;
                    V3 l = vscale(to_l, 1.0f / sqrtf(d2));
                    float ndl = vdot(st->normal, l);
                    if (ndl <= 0.0f) continue;
                    c = vadd(c, vscale(vmul(vscale(st->material.albedo, ndl), light->color),
                                       light->power / (d2 + 1.0f)));
                }
                int out = idx * 3;
                rgb[out] = toByteFast(c.x);
                rgb[out + 1] = toByteFast(c.y);
                rgb[out + 2] = toByteFast(c.z);
            }
        }
    }
    free(row_triangles);
    free(row_cursor);
    free(row_offsets);
    free(row_counts);
    free(screen);
    free(depth);
}
#endif

#if defined(__AVX2__)
static RT_UNUSED unsigned char toByteLut(float x) {
    if (!(x > 0.0f)) return 0;
    if (x >= 1.0f) return 255;
    return fast_gamma_lut[(int)(x * 4095.0f)];
}
#endif

static void render_camera_basis(const Scene *s, int width, int height,
                                V3 *fwd, V3 *right, V3 *up, float *aspect, float *tanH) {
    *fwd = vnorm(vsub(s->target, s->camera));
    *right = vnorm(vcross(*fwd, v3(0, 1, 0)));
    *up = vcross(*right, *fwd);
    *aspect = (float)width / (float)height;
    *tanH = tanf(s->fov * 3.14159265f / 360.0f);
}

static V3 camera_ray(V3 fwd, V3 right, V3 up,
                     float aspect, float tanH, int x, int y, int width, int height) {
    float u = ((float)x + 0.5f) / (float)width * 2.0f - 1.0f;
    float v = 1.0f - ((float)y + 0.5f) / (float)height * 2.0f;
    return vnorm(vadd(fwd, vadd(vscale(right, u * aspect * tanH), vscale(up, v * tanH))));
}

typedef struct {
    V3 color;
    int object;
} RtRenderSample;

int rt_render_blocked(const Scene *s, unsigned char *rgb, int width, int height,
                      int block_size, int shadow_samples, int max_depth) {
    if (!s || !rgb || width <= 0 || height <= 0) return 0;
    if (block_size < 1) block_size = 1;
    V3 fwd, right, up;
    float aspect, tanH;
    render_camera_basis(s, width, height, &fwd, &right, &up, &aspect, &tanH);
    int grid_w = (width + block_size - 1) / block_size;
    int grid_h = (height + block_size - 1) / block_size;
    RtRenderSample *samples = (RtRenderSample *)malloc((size_t)grid_w * grid_h * sizeof *samples);
    if (!samples) return 0;
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static) if(block_size > 1)
#endif
    for (int gy = 0; gy < grid_h; gy++) {
        for (int gx = 0; gx < grid_w; gx++) {
            int bx = gx * block_size;
            int by = gy * block_size;
            V3 rd = camera_ray(fwd, right, up, aspect, tanH,
                               bx + block_size / 2, by + block_size / 2,
                               width, height);
            int object = -1;
            samples[gy * grid_w + gx].color = radiance_ex(
                s, s->camera, rd, 0, shadow_samples, max_depth, &object);
            samples[gy * grid_w + gx].object = object;
        }
    }
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(block_size > 1)
#endif
    for (int y = 0; y < height; y++) {
        /* Sample positions are at the centers of the traced blocks. */
        float sy = ((float)y + 0.5f) / (float)block_size - 0.5f;
        int y0 = (int)floorf(sy);
        float ty = sy - (float)y0;
        if (y0 < 0) { y0 = 0; ty = 0.0f; }
        if (y0 >= grid_h - 1) { y0 = grid_h - 1; ty = 0.0f; }
        int y1 = y0 + 1 < grid_h ? y0 + 1 : y0;
        for (int x = 0; x < width; x++) {
            float sx = ((float)x + 0.5f) / (float)block_size - 0.5f;
            int x0 = (int)floorf(sx);
            float tx = sx - (float)x0;
            if (x0 < 0) { x0 = 0; tx = 0.0f; }
            if (x0 >= grid_w - 1) { x0 = grid_w - 1; tx = 0.0f; }
            int x1 = x0 + 1 < grid_w ? x0 + 1 : x0;
            RtRenderSample s00 = samples[y0 * grid_w + x0];
            RtRenderSample s10 = samples[y0 * grid_w + x1];
            RtRenderSample s01 = samples[y1 * grid_w + x0];
            RtRenderSample s11 = samples[y1 * grid_w + x1];
            V3 c;
            if (s00.object == s10.object && s00.object == s01.object &&
                s00.object == s11.object) {
                V3 top = vadd(vscale(s00.color, 1.0f - tx), vscale(s10.color, tx));
                V3 bottom = vadd(vscale(s01.color, 1.0f - tx), vscale(s11.color, tx));
                c = vadd(vscale(top, 1.0f - ty), vscale(bottom, ty));
            } else {
                int nx = tx < 0.5f ? x0 : x1;
                int ny = ty < 0.5f ? y0 : y1;
                c = samples[ny * grid_w + nx].color;
            }
            int i = (y * width + x) * 3;
            rgb[i + 0] = toByteFast(c.x);
            rgb[i + 1] = toByteFast(c.y);
            rgb[i + 2] = toByteFast(c.z);
        }
    }

    free(samples);
    return 1;
}

int rt_render_exact(const Scene *s, unsigned char *rgb, int width, int height,
                    int shadow_samples, int max_depth) {
    if (!s || !rgb || width <= 0 || height <= 0) return 0;
    V3 *hdr = (V3 *)malloc((size_t)width * height * sizeof *hdr);
    if (!hdr) return 0;
    V3 fwd, right, up;
    float aspect, tanH;
    render_camera_basis(s, width, height, &fwd, &right, &up, &aspect, &tanH);
    if (shadow_samples < 1) shadow_samples = 1;
    if (max_depth < 0) max_depth = 0;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            V3 rd = camera_ray(fwd, right, up, aspect, tanH,
                               x, y, width, height);
             unsigned int rng = pixel_seed(x, y);
             V3 c = path_radiance(s, s->camera, rd, &rng, max_depth, NULL);
             hdr[y * width + x] = c;
        }
    }
    int ok = postprocess_hdr(hdr, rgb, width, height);
    free(hdr);
    return ok;
}

#if defined(__AVX2__)
static __m256 avx_abs(__m256 a) {
    return _mm256_andnot_ps(_mm256_set1_ps(-0.0f), a);
}

static __m256 avx_min3(__m256 a, __m256 b, __m256 c) {
    return _mm256_min_ps(_mm256_min_ps(a, b), c);
}

static __m256 avx_max3(__m256 a, __m256 b, __m256 c) {
    return _mm256_max_ps(_mm256_max_ps(a, b), c);
}

static void avx_select(__m256 mask, __m256 *dst, __m256 value) {
    *dst = _mm256_blendv_ps(*dst, value, mask);
}

static RT_UNUSED void avx_intersect_small(const Scene *s, V3 ro, __m256 dx, __m256 dy,
                                __m256 dz, __m256 *best_t, __m256 *out_nx,
                                __m256 *out_ny, __m256 *out_nz, int ids[8]) {
    __m256 inf = _mm256_set1_ps(1e30f);
    __m256 best = inf;
    __m256 nx = _mm256_setzero_ps(), ny = _mm256_setzero_ps(), nz = _mm256_setzero_ps();
    int id_lanes[8];
    for (int lane = 0; lane < 8; lane++) id_lanes[lane] = -1;

    for (int oi = 0; oi < s->count; oi++) {
        const Object *o = &s->objects[oi];
        __m256 t = inf, hit_mask = _mm256_setzero_ps();
        __m256 hn_x = _mm256_setzero_ps(), hn_y = _mm256_setzero_ps(), hn_z = _mm256_setzero_ps();
        if (o->shape == RT_PLANE) {
            __m256 oy = _mm256_set1_ps(o->geometry.plane.y - ro.y);
            t = _mm256_div_ps(oy, dy);
            hit_mask = _mm256_and_ps(_mm256_cmp_ps(avx_abs(dy), _mm256_set1_ps(1e-9f), _CMP_GT_OQ),
                                     _mm256_cmp_ps(t, _mm256_set1_ps(RT_EPSILON), _CMP_GT_OQ));
            hn_y = _mm256_blendv_ps(_mm256_set1_ps(1.0f), _mm256_set1_ps(-1.0f),
                                    _mm256_cmp_ps(dy, _mm256_setzero_ps(), _CMP_GT_OQ));
        } else if (o->shape == RT_SPHERE) {
            __m256 cx = _mm256_set1_ps(o->geometry.sphere.center.x - ro.x);
            __m256 cy = _mm256_set1_ps(o->geometry.sphere.center.y - ro.y);
            __m256 cz = _mm256_set1_ps(o->geometry.sphere.center.z - ro.z);
            __m256 b = _mm256_add_ps(_mm256_add_ps(cx, cy), cz);
            b = _mm256_mul_ps(b, _mm256_set1_ps(0.0f));
            b = _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(ro.x - o->geometry.sphere.center.x), dx),
                              _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(ro.y - o->geometry.sphere.center.y), dy),
                                            _mm256_mul_ps(_mm256_set1_ps(ro.z - o->geometry.sphere.center.z), dz)));
            __m256 c = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(cx, cx), _mm256_mul_ps(cy, cy)), _mm256_mul_ps(cz, cz));
            c = _mm256_sub_ps(c, _mm256_set1_ps(o->geometry.sphere.radius * o->geometry.sphere.radius));
            __m256 disc = _mm256_sub_ps(_mm256_mul_ps(b, b), c);
            __m256 valid = _mm256_cmp_ps(disc, _mm256_setzero_ps(), _CMP_GE_OQ);
            __m256 root = _mm256_sqrt_ps(_mm256_max_ps(disc, _mm256_setzero_ps()));
            t = _mm256_sub_ps(_mm256_sub_ps(_mm256_setzero_ps(), b), root);
            hit_mask = _mm256_and_ps(valid, _mm256_cmp_ps(t, _mm256_set1_ps(RT_EPSILON), _CMP_GT_OQ));
            __m256 px = _mm256_add_ps(_mm256_set1_ps(ro.x), _mm256_mul_ps(dx, t));
            __m256 py = _mm256_add_ps(_mm256_set1_ps(ro.y), _mm256_mul_ps(dy, t));
            __m256 pz = _mm256_add_ps(_mm256_set1_ps(ro.z), _mm256_mul_ps(dz, t));
            __m256 invr = _mm256_set1_ps(1.0f / o->geometry.sphere.radius);
            hn_x = _mm256_mul_ps(_mm256_sub_ps(px, _mm256_set1_ps(o->geometry.sphere.center.x)), invr);
            hn_y = _mm256_mul_ps(_mm256_sub_ps(py, _mm256_set1_ps(o->geometry.sphere.center.y)), invr);
            hn_z = _mm256_mul_ps(_mm256_sub_ps(pz, _mm256_set1_ps(o->geometry.sphere.center.z)), invr);
        } else if (o->shape == RT_BOX) {
            __m256 invx = _mm256_div_ps(_mm256_set1_ps(1.0f), dx);
            __m256 invy = _mm256_div_ps(_mm256_set1_ps(1.0f), dy);
            __m256 invz = _mm256_div_ps(_mm256_set1_ps(1.0f), dz);
            __m256 ax0 = _mm256_mul_ps(_mm256_sub_ps(_mm256_set1_ps(o->geometry.box.min.x), _mm256_set1_ps(ro.x)), invx);
            __m256 ax1 = _mm256_mul_ps(_mm256_sub_ps(_mm256_set1_ps(o->geometry.box.max.x), _mm256_set1_ps(ro.x)), invx);
            __m256 ay0 = _mm256_mul_ps(_mm256_sub_ps(_mm256_set1_ps(o->geometry.box.min.y), _mm256_set1_ps(ro.y)), invy);
            __m256 ay1 = _mm256_mul_ps(_mm256_sub_ps(_mm256_set1_ps(o->geometry.box.max.y), _mm256_set1_ps(ro.y)), invy);
            __m256 az0 = _mm256_mul_ps(_mm256_sub_ps(_mm256_set1_ps(o->geometry.box.min.z), _mm256_set1_ps(ro.z)), invz);
            __m256 az1 = _mm256_mul_ps(_mm256_sub_ps(_mm256_set1_ps(o->geometry.box.max.z), _mm256_set1_ps(ro.z)), invz);
            __m256 axn = _mm256_min_ps(ax0, ax1), axf = _mm256_max_ps(ax0, ax1);
            __m256 ayn = _mm256_min_ps(ay0, ay1), ayf = _mm256_max_ps(ay0, ay1);
            __m256 azn = _mm256_min_ps(az0, az1), azf = _mm256_max_ps(az0, az1);
            __m256 near = avx_max3(axn, ayn, azn), far = avx_min3(axf, ayf, azf);
            t = near;
            hit_mask = _mm256_and_ps(_mm256_cmp_ps(near, far, _CMP_LE_OQ),
                                     _mm256_cmp_ps(near, _mm256_set1_ps(RT_EPSILON), _CMP_GT_OQ));
            __m256 choose_x = _mm256_cmp_ps(near, axn, _CMP_EQ_OQ);
            __m256 choose_y = _mm256_andnot_ps(choose_x, _mm256_cmp_ps(near, ayn, _CMP_EQ_OQ));
            __m256 sign_x = _mm256_blendv_ps(_mm256_set1_ps(1.0f), _mm256_set1_ps(-1.0f), _mm256_cmp_ps(dx, _mm256_setzero_ps(), _CMP_GT_OQ));
            __m256 sign_y = _mm256_blendv_ps(_mm256_set1_ps(1.0f), _mm256_set1_ps(-1.0f), _mm256_cmp_ps(dy, _mm256_setzero_ps(), _CMP_GT_OQ));
            __m256 sign_z = _mm256_blendv_ps(_mm256_set1_ps(1.0f), _mm256_set1_ps(-1.0f), _mm256_cmp_ps(dz, _mm256_setzero_ps(), _CMP_GT_OQ));
            hn_x = _mm256_and_ps(choose_x, sign_x);
            hn_y = _mm256_and_ps(choose_y, sign_y);
            hn_z = _mm256_andnot_ps(_mm256_or_ps(choose_x, choose_y), sign_z);
        }
        hit_mask = _mm256_and_ps(hit_mask, _mm256_cmp_ps(t, best, _CMP_LT_OQ));
        avx_select(hit_mask, &best, t);
        avx_select(hit_mask, &nx, hn_x);
        avx_select(hit_mask, &ny, hn_y);
        avx_select(hit_mask, &nz, hn_z);
        int mask = _mm256_movemask_ps(hit_mask);
        for (int lane = 0; lane < 8; lane++) if (mask & (1 << lane)) id_lanes[lane] = oi;
    }
    *best_t = best; *out_nx = nx; *out_ny = ny; *out_nz = nz;
    for (int lane = 0; lane < 8; lane++) ids[lane] = id_lanes[lane];
}

int rt_render_exact_fast(const Scene *s, unsigned char *rgb, int width, int height,
                         int shadow_samples, int max_depth) {
    (void)shadow_samples; (void)max_depth;
    if (!s || !rgb || width <= 0 || height <= 0) return 0;
    if (s->mesh_count > 0 && s->render_meshes) {
        Scene *background = (Scene *)malloc(sizeof *background);
        if (!background) return 0;
        *background = *s;
        background->render_meshes = 0;
        int ok = rt_render_exact_fast(background, rgb, width, height, shadow_samples, max_depth);
        free(background);
        raster_meshes(s, rgb, width, height);
        return ok;
    }
    init_fast_gamma();
    V3 fwd, right, up; float aspect, tanH;
    render_camera_basis(s, width, height, &fwd, &right, &up, &aspect, &tanH);
#pragma omp parallel for schedule(static)
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x += 8) {
            float xs[8];
            for (int lane = 0; lane < 8; lane++) xs[lane] = (float)(x + lane) + 0.5f;
            __m256 u = _mm256_sub_ps(_mm256_mul_ps(_mm256_loadu_ps(xs), _mm256_set1_ps(2.0f / width)), _mm256_set1_ps(1.0f));
            __m256 v = _mm256_set1_ps(1.0f - ((float)y + 0.5f) / height * 2.0f);
            __m256 dx = _mm256_add_ps(_mm256_set1_ps(fwd.x), _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(right.x * aspect * tanH), u), _mm256_mul_ps(_mm256_set1_ps(up.x * tanH), v)));
            __m256 dy = _mm256_add_ps(_mm256_set1_ps(fwd.y), _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(right.y * aspect * tanH), u), _mm256_mul_ps(_mm256_set1_ps(up.y * tanH), v)));
            __m256 dz = _mm256_add_ps(_mm256_set1_ps(fwd.z), _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(right.z * aspect * tanH), u), _mm256_mul_ps(_mm256_set1_ps(up.z * tanH), v)));
            __m256 len = _mm256_sqrt_ps(_mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(dx, dx), _mm256_mul_ps(dy, dy)), _mm256_mul_ps(dz, dz)));
            dx = _mm256_div_ps(dx, len); dy = _mm256_div_ps(dy, len); dz = _mm256_div_ps(dz, len);
            __m256 t, nx, ny, nz; int ids[8];
            avx_intersect_small(s, s->camera, dx, dy, dz, &t, &nx, &ny, &nz, ids);
            float td[8], nxd[8], nyd[8], nzd[8], dxd[8], dyd[8], dzd[8];
            _mm256_storeu_ps(td, t); _mm256_storeu_ps(nxd, nx); _mm256_storeu_ps(nyd, ny); _mm256_storeu_ps(nzd, nz);
            _mm256_storeu_ps(dxd, dx); _mm256_storeu_ps(dyd, dy); _mm256_storeu_ps(dzd, dz);
            for (int lane = 0; lane < 8 && x + lane < width; lane++) {
                V3 rd = v3(dxd[lane], dyd[lane], dzd[lane]);
                V3 n = v3(nxd[lane], nyd[lane], nzd[lane]);
                V3 c = small_shade_packet_hit(s, s->camera, rd, td[lane], n, ids[lane]);
                int i = (y * width + x + lane) * 3;
                rgb[i] = toByteLut(c.x);
                rgb[i + 1] = toByteLut(c.y);
                rgb[i + 2] = toByteLut(c.z);
            }
        }
    }
    return 1;
}
#else
int rt_render_exact_fast(const Scene *s, unsigned char *rgb, int width, int height,
                         int shadow_samples, int max_depth) {
    return rt_render_exact(s, rgb, width, height, shadow_samples, max_depth);
}
#endif

static float rgb_luma8(const unsigned char *rgb, int index) {
    return 0.2126f * rgb[index + 0] + 0.7152f * rgb[index + 1] +
           0.0722f * rgb[index + 2];
}

static void upscale_adaptive_rgb(const unsigned char *coarse, unsigned char *output,
                                 int coarse_w, int coarse_h, int width, int height) {
    for (int y = 0; y < height; y++) {
        float source_y = ((float)y + 0.5f) * coarse_h / (float)height - 0.5f;
        int y0 = (int)floorf(source_y);
        float fy = source_y - y0;
        if (y0 < 0) { y0 = 0; fy = 0.0f; }
        if (y0 >= coarse_h - 1) { y0 = coarse_h - 1; fy = 0.0f; }
        int y1 = y0 + 1 < coarse_h ? y0 + 1 : y0;
        for (int x = 0; x < width; x++) {
            float source_x = ((float)x + 0.5f) * coarse_w / (float)width - 0.5f;
            int x0 = (int)floorf(source_x);
            float fx = source_x - x0;
            if (x0 < 0) { x0 = 0; fx = 0.0f; }
            if (x0 >= coarse_w - 1) { x0 = coarse_w - 1; fx = 0.0f; }
            int x1 = x0 + 1 < coarse_w ? x0 + 1 : x0;
            int ids[4] = {
                (y0 * coarse_w + x0) * 3, (y0 * coarse_w + x1) * 3,
                (y1 * coarse_w + x0) * 3, (y1 * coarse_w + x1) * 3
            };
            float weights[4] = {
                (1.0f - fx) * (1.0f - fy), fx * (1.0f - fy),
                (1.0f - fx) * fy, fx * fy
            };
            float center = rgb_luma8(coarse, ids[0]);
            float sum = 0.0f;
            V3 color = v3(0, 0, 0);
            for (int i = 0; i < 4; i++) {
                float edge = fabsf(rgb_luma8(coarse, ids[i]) - center) / 255.0f;
                float guide_weight = 1.0f / (1.0f + edge * 8.0f);
                float weight = weights[i] * guide_weight;
                color = vadd(color, vscale(v3(coarse[ids[i] + 0], coarse[ids[i] + 1],
                                               coarse[ids[i] + 2]), weight));
                sum += weight;
            }
            if (sum > 0.0f) color = vscale(color, 1.0f / sum);
            int output_index = (y * width + x) * 3;
            output[output_index + 0] = (unsigned char)ffmaxf_(0.0f, ffminf_(color.x, 255.0f));
            output[output_index + 1] = (unsigned char)ffmaxf_(0.0f, ffminf_(color.y, 255.0f));
            output[output_index + 2] = (unsigned char)ffmaxf_(0.0f, ffminf_(color.z, 255.0f));
        }
    }
}

int rt_render_adaptive_fast(const Scene *s, unsigned char *rgb, int width, int height,
                            int spp_min, int spp_max) {
    (void)spp_min;
    (void)spp_max;
    if (!s || !rgb || width <= 0 || height <= 0) return 0;
    int coarse_w = width / 4;
    int coarse_h = height / 4;
    if (coarse_w < 1) coarse_w = 1;
    if (coarse_h < 1) coarse_h = 1;
    unsigned char *coarse = (unsigned char *)malloc((size_t)coarse_w * coarse_h * 3);
    if (!coarse) return 0;
    int ok = rt_render_exact_fast(s, coarse, coarse_w, coarse_h, 1, 1);
    if (ok) upscale_adaptive_rgb(coarse, rgb, coarse_w, coarse_h, width, height);
    free(coarse);
    return ok;
}

int rt_render_rows(const Scene *s, unsigned char *rgb, int width, int height,
                   int samples, int y0, int y1, int shadow_samples, int max_depth) {
    if (!s || !rgb || width <= 0 || height <= 0) return 0;
    if (samples != 1 && samples != 4 && samples != 16) return 0;
    if (y0 < 0) y0 = 0;
    if (y1 > height) y1 = height;
    if (y0 >= y1) return 1;
    if (shadow_samples < 1) shadow_samples = 1;
    if (max_depth < 0) max_depth = 0;
    int grid = samples == 1 ? 1 : (samples == 4 ? 2 : 4);
    float inv = 1.0f / (float)(grid * grid);

    V3 fwd = vnorm(vsub(s->target, s->camera));
    V3 right = vnorm(vcross(fwd, v3(0, 1, 0)));
    V3 up = vcross(right, fwd);
    float aspect = (float)width / (float)height;
    float tanH = tanf(s->fov * 3.14159265f / 360.0f);

    for (int y = y0; y < y1; y++) {
        for (int x = 0; x < width; x++) {
            V3 acc = v3(0, 0, 0);
            for (int sy = 0; sy < grid; sy++) {
                for (int sx = 0; sx < grid; sx++) {
                    float fx = (float)x + (sx + 0.5f) / grid;
                    float fy = (float)y + (sy + 0.5f) / grid;
                    float u = fx / width * 2.0f - 1.0f;
                    float v = 1.0f - fy / height * 2.0f;
                    V3 rd = vnorm(vadd(fwd, vadd(
                        vscale(right, u * aspect * tanH),
                        vscale(up, v * tanH))));
                    V3 c = radiance_ex(s, s->camera, rd, 0, shadow_samples, max_depth, NULL);
                    if (!(c.x > -1.0f && c.x < 10.0f) ||
                        !(c.y > -1.0f && c.y < 10.0f) ||
                        !(c.z > -1.0f && c.z < 10.0f)) return 0;
                    acc = vadd(acc, vscale(c, inv));
                }
            }
            int i = (y * width + x) * 3;
            rgb[i + 0] = toByte(acc.x);
            rgb[i + 1] = toByte(acc.y);
            rgb[i + 2] = toByte(acc.z);
        }
    }
    return 1;
}

int rt_render(const Scene *s, unsigned char *rgb, int width, int height, int samples) {
    if (!s || !rgb || width <= 0 || height <= 0) return 0;
    if (samples != 1 && samples != 4 && samples != 16) return 0;
    int grid = samples == 1 ? 1 : (samples == 4 ? 2 : 4);
    float inv = 1.0f / (float)(grid * grid);
    V3 *hdr = (V3 *)malloc((size_t)width * height * sizeof *hdr);
    if (!hdr) return 0;

    V3 fwd, right, up;
    float aspect, tanH;
    render_camera_basis(s, width, height, &fwd, &right, &up, &aspect, &tanH);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            V3 acc = v3(0, 0, 0);
            for (int sy = 0; sy < grid; sy++) {
                for (int sx = 0; sx < grid; sx++) {
                    float fx = (float)x + ((float)sx + 0.5f) / grid;
                    float fy = (float)y + ((float)sy + 0.5f) / grid;
                    float u = fx / width * 2.0f - 1.0f;
                    float v = 1.0f - fy / height * 2.0f;
                    V3 rd = vnorm(vadd(fwd, vadd(
                        vscale(right, u * aspect * tanH),
                        vscale(up, v * tanH))));
                    V3 c = radiance_ex(s, s->camera, rd, 0, 5, 2, NULL);
                    if (!isfinite(c.x) || !isfinite(c.y) || !isfinite(c.z)) {
                        free(hdr);
                        return 0;
                    }
                    acc = vadd(acc, vscale(c, inv));
                }
            }
            hdr[y * width + x] = acc;
        }
    }
    int ok = postprocess_hdr(hdr, rgb, width, height);
    free(hdr);
    return ok;
}

void scene_block(Scene *s);
void scene_world(Scene *s);
/* scene builders live in tracer_scene.c */
