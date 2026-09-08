/* OpenCL port of the CPU estimator in tracer.c so the GPU frame matches:
 *   - path_radiance(): emission + ambient sky term + MIS direct light
 *     (GGX/Smith/Schlick BRDF, stochastic light-disk sampling) + BSDF
 *     sampling with Russian roulette, max_depth 4.
 *   - sky_color(): gradient + sun glow (glow^128) + sun disk (dot^9000).
 *   - material_albedo(): procedural textures (checker/grass/dirt/stone/
 *     wall/roof) + legacy ground-plane checker.
 *   - Sobol/Owen RNG with the same pixel seeds and draw order as the CPU.
 * Output is linear HDR; the host runs the exact CPU post chain
 * (exposure, bloom, Reinhard, sRGB) via rt_postprocess_hdr(). */

#define PI_F 3.14159265f
#define EPS 1e-5f
#define MAX_DEPTH 4

/* Per-frame scalar block (see OCL_FRAME_* in gpu_opencl.c). Every value the
 * trace kernels need that changes per frame travels through one buffer, so
 * kernel args are bound once and SetKernelArg never runs in the frame loop
 * (it costs ~3 ms per call on 51-arg kernels with the Gen9 driver). */
#define F_CAM 0
#define F_FWD 4
#define F_RIGHT 8
#define F_UP 12
#define F_ASPECT 16
#define F_W 20
#define F_H 21
#define F_SPP 22
#define F_SAMPLE_BASE 23
#define F_COUNTS 24
#define F_TRI 28
#define F_LIGHTS 29
#define F_FOG 32
#define F_SUN_DIR 36
#define F_SUN_COL 40
#define F_BMAT 44
#define F_BEMI 48
#define F_ACCUM 30
#define FRAME_FLOATS 52

struct Hit { float t; float3 n; int kind; int idx; };

float clamp01(float x) { return clamp(x, 0.0f, 1.0f); }
float lum(float3 c) { return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z; }

/* ---- RNG: Sobol/Owen, identical to tracer.c ---- */
uint sobol_owen(uint index, uint dimension) {
    uint value = 0;
    uint direction = 0x80000000u >> (dimension & 7u);
    for (uint bit = 0; index != 0u && bit < 32u; bit++, index >>= 1) {
        if ((index & 1u) != 0u) value ^= direction;
        direction = (direction >> 1) ^ (0x80200003u & (uint)(-(int)(direction & 1u)));
    }
    value ^= value * 0x3d20adeau;
    value ^= value >> 11;
    value *= 0x05526c56u;
    value ^= value >> 13;
    return value;
}
float rng_float(uint *state) {
    uint dimension = (*state) & 7u;
    uint index = (*state) >> 3;
    *state += 1u;
    return ((float)sobol_owen(index + 1u, dimension) + 0.5f) / 4294967296.0f;
}
uint pixel_seed(int x, int y) {
    uint seed = (uint)(x + 1) * 0x9e3779b9u;
    seed ^= (uint)(y + 1) * 0x85ebca6bu;
    seed ^= 0xc2b2ae35u;
    return ((seed % 65521u) << 3) | 0u;
}

/* ---- BRDF ---- */
float3 fresnel_schlick(float3 f0, float cos_theta) {
    float m = 1.0f - clamp01(cos_theta);
    float factor = m * m * m * m * m;
    return f0 + (1.0f - f0) * factor;
}
float ggx_d(float no_h, float rough) {
    float a = rough * rough, a2 = a * a, n2 = no_h * no_h;
    float q = n2 * (a2 - 1.0f) + 1.0f;
    return a2 / (PI_F * q * q);
}
float smith_g1(float no_v, float rough) {
    float k = (rough + 1.0f) * (rough + 1.0f) * 0.125f;
    return no_v / (no_v * (1.0f - k) + k);
}
float smith_vis(float no_l, float no_v, float rough) {
    return smith_g1(no_l, rough) * smith_g1(no_v, rough);
}
float roughness_of(float shininess) {
    return clamp(sqrt(2.0f / (shininess + 2.0f)), 0.045f, 1.0f);
}
void tangent_basis(float3 n, float3 *u, float3 *v) {
    float3 helper = fabs(n.y) < 0.9f ? (float3)(0.0f, 1.0f, 0.0f) : (float3)(1.0f, 0.0f, 0.0f);
    *u = normalize(cross(helper, n));
    *v = cross(n, *u);
}
void brdf_eval(float3 albedo, float3 n, float3 view, float3 light_dir,
               float reflection, float shininess, float *pdf, float3 *out_f) {
    float no_l = fmax(dot(n, light_dir), 0.0f);
    float no_v = fmax(dot(n, view), 0.0f);
    if (no_l <= 0.0f || no_v <= 0.0f) { *pdf = 0.0f; *out_f = (float3)(0.0f); return; }
    float3 h = normalize(light_dir + view);
    float no_h = fmax(dot(n, h), 0.0f);
    float vo_h = fmax(dot(view, h), 0.0f);
    float rough = roughness_of(shininess);
    float spec_probability = clamp01(0.25f + 0.70f * reflection);
    float3 f0 = 0.04f * (1.0f - reflection) + albedo * reflection;
    float3 f = fresnel_schlick(f0, vo_h);
    float d = ggx_d(no_h, rough);
    float g = smith_vis(no_l, no_v, rough);
    float3 spec = f * (d * g / fmax(4.0f * no_l * no_v, 1e-5f));
    float3 diff = albedo * (1.0f - f) / PI_F;
    float diffuse_pdf = no_l / PI_F;
    float spec_pdf = d * no_h / fmax(4.0f * vo_h, 1e-5f);
    *pdf = (1.0f - spec_probability) * diffuse_pdf + spec_probability * spec_pdf;
    *out_f = diff + spec;
}
float3 sample_cosine(float3 n, uint *rng, float *pdf) {
    float r = sqrt(rng_float(rng));
    float phi = 2.0f * PI_F * rng_float(rng);
    float z = sqrt(fmax(1.0f - r * r, 0.0f));
    float3 u, v;
    tangent_basis(n, &u, &v);
    float3 d = u * (r * cos(phi)) + v * (r * sin(phi)) + n * z;
    *pdf = z / PI_F;
    return normalize(d);
}
float3 sample_ggx(float3 n, float rough, uint *rng, float *pdf) {
    float alpha = rough * rough;
    float phi = 2.0f * PI_F * rng_float(rng);
    float u = rng_float(rng);
    float cos_theta = sqrt((1.0f - u) / (1.0f + (alpha * alpha - 1.0f) * u));
    float sin_theta = sqrt(fmax(1.0f - cos_theta * cos_theta, 0.0f));
    float3 t, b;
    tangent_basis(n, &t, &b);
    float3 h = normalize(t * (sin_theta * cos(phi)) + b * (sin_theta * sin(phi)) + n * cos_theta);
    *pdf = ggx_d(fmax(dot(n, h), 0.0f), rough) / fmax(4.0f * fabs(dot(h, n)), 1e-5f);
    return h;
}
float3 sample_bsdf(float3 albedo, float3 n, float3 view, float reflection,
                   float shininess, uint *rng, float3 *direction, float *pdf) {
    float rough = roughness_of(shininess);
    float spec_probability = clamp01(0.25f + 0.70f * reflection);
    if (rng_float(rng) < spec_probability) {
        float3 h = sample_ggx(n, rough, rng, pdf);
        *direction = normalize(h * (2.0f * dot(view, h)) - view);
    } else {
        *direction = sample_cosine(n, rng, pdf);
    }
    if (dot(n, *direction) <= 0.0f) { *pdf = 0.0f; return (float3)(0.0f); }
    float3 f;
    brdf_eval(albedo, n, view, *direction, reflection, shininess, pdf, &f);
    return f;
}

/* ---- procedural albedo (material_albedo port) ---- */
float tex_hash(int x, int y, int z) {
    uint n = (uint)x * 374761393u;
    n += (uint)y * 668265263u;
    n += (uint)z * 2147483647u;
    n = (n ^ (n >> 13)) * 1274126177u;
    n ^= n >> 16;
    return (float)(n & 0xffffu) / 65535.0f;
}
float3 material_albedo(float3 base_albedo, float4 texA, float4 texB, float3 p) {
    int ttype = (int)texA.x;
    if (ttype == 0) return base_albedo;
    float scale = texA.y > 0.0f ? texA.y : 1.0f;
    int ix = (int)floor(p.x * scale);
    int iy = (int)floor(p.y * scale);
    int iz = (int)floor(p.z * scale);
    float value = tex_hash(ix, iy, iz);
    float3 alternate = texB.xyz;
    if (ttype == 1) {                       /* CHECKER */
        value = ((ix + iz) & 1) ? 1.0f : 0.0f;
    } else if (ttype == 2) {                /* GRASS */
        value = 0.72f + 0.28f * value;
    } else if (ttype == 3) {                /* DIRT */
        value = 0.68f + 0.32f * value;
    } else if (ttype == 4) {                /* STONE */
        value = 0.70f + 0.30f * value;
    } else if (ttype == 5) {                /* WALL */
        value = 0.82f + 0.18f * value;
    } else if (ttype == 6) {                /* ROOF */
        value = (((ix + iz * 3) & 3) == 0) ? 0.48f : (0.78f + 0.22f * value);
    }
    value = 1.0f - texA.z + texA.z * value;
    return base_albedo * value + alternate * (1.0f - value);
}

/* ---- geometry ---- */
float2 sph_hit(const float4 s, const float3 ro, const float3 rd) {
    float3 oc = ro - s.xyz;
    float b = dot(oc, rd);
    float c = dot(oc, oc) - s.w * s.w;
    float disc = b * b - c;
    if (disc < 0.0f) return (float2)(-1.0f, -1.0f);
    float sq = sqrt(disc);
    float t0 = -b - sq, t1 = -b + sq;
    if (t0 > EPS) return (float2)(t0, t1);
    if (t1 > EPS) return (float2)(t1, t0);
    return (float2)(-1.0f, -1.0f);
}
float2 box_hit(const float4 mn, const float4 mx, const float3 ro, const float3 rd) {
    float3 inv = 1.0f / rd;
    float3 t0 = (mn.xyz - ro) * inv;
    float3 t1 = (mx.xyz - ro) * inv;
    float3 tn = fmin(t0, t1), tf = fmax(t0, t1);
    float near = fmax(fmax(tn.x, tn.y), tn.z);
    float far = fmin(fmin(tf.x, tf.y), tf.z);
    if (near > far || far < EPS) return (float2)(-1.0f, -1.0f);
    return (float2)(near > EPS ? near : far, near);
}
float2 cyl_hit(const float4 base, const float2 ht, const float3 ro, const float3 rd) {
    float r = base.w;
    float2 d = ro.xz - base.xz;
    float a = rd.x * rd.x + rd.z * rd.z;
    float b = 2.0f * dot(d, rd.xz);
    float c = dot(d, d) - r * r;
    float best = 1e30f;
    int found = 0, side = 0;
    if (a > 1e-12f) {
        float disc = b * b - 4.0f * a * c;
        if (disc >= 0.0f) {
            float q = sqrt(disc);
            float t0 = (-b - q) / (2.0f * a), t1 = (-b + q) / (2.0f * a);
            if (t0 > EPS && t0 < best) { float y = ro.y + rd.y * t0; if (y >= base.y && y <= ht.x) { best = t0; found = 1; side = 1; } }
            if (t1 > EPS && t1 < best) { float y = ro.y + rd.y * t1; if (y >= base.y && y <= ht.x) { best = t1; found = 1; side = 1; } }
        }
    }
    if (fabs(rd.y) > 1e-9f) {
        float t = (base.y - ro.y) / rd.y;
        float2 q = ro.xz + rd.xz * t - base.xz;
        if (t > EPS && dot(q, q) <= r * r && t < best) { best = t; found = 1; side = 0; }
        t = (ht.x - ro.y) / rd.y;
        q = ro.xz + rd.xz * t - base.xz;
        if (t > EPS && dot(q, q) <= r * r && t < best) { best = t; found = 1; side = 0; }
    }
    return found ? (float2)(best, (float)side) : (float2)(-1.0f, 0.0f);
}
float plane_hit(const float y, const float3 ro, const float3 rd) {
    if (fabs(rd.y) < 1e-9f) return -1.0f;
    float t = (y - ro.y) / rd.y;
    return t > EPS ? t : -1.0f;
}
float3 tri_fetch(__global const float *tris, const int tri, const int k) {
    return vload3(tri * 3 + k, tris);
}
float tri_hit(__global const float *tris, const int tri, const float3 ro, const float3 rd) {
    float3 v0 = tri_fetch(tris, tri, 0);
    float3 e1 = tri_fetch(tris, tri, 1);
    float3 e2 = tri_fetch(tris, tri, 2);
    float3 p = cross(rd, e2);
    float det = dot(e1, p);
    if (fabs(det) < 1e-8f) return -1.0f;
    float inv = 1.0f / det;
    float3 tv = ro - v0;
    float u = dot(tv, p) * inv;
    if (u < 0.0f || u > 1.0f) return -1.0f;
    float3 q = cross(tv, e1);
    float v = dot(rd, q) * inv;
    if (v < 0.0f || u + v > 1.0f) return -1.0f;
    float t = dot(e2, q) * inv;
    return t > EPS ? t : -1.0f;
}

/* ---- uniform grid over the bunny triangles (built once on the host) ----
 * grid_a:  (origin.xyz, cell_size)
 * grid_dims: (nx, ny, nz, 0);  CSR cells: grid_off[n+1], grid_tri[].
 * Returns triangle index of the closest hit within max_t, -1 if none.
 * When occlusion_only, returns early on any hit. */
int grid_traverse(const float3 ro, const float3 rd, const float max_t,
                  const bool occlusion_only, float *best_t_out,
                  __global const float4 *grid_a, __global const uint4 *grid_dims,
                  __global const uint *grid_off, __global const uint *grid_tri,
                  __global const float *tris) {
    *best_t_out = 1e30f;
    if (grid_dims[0].x == 0u) return -1;
    float3 origin = grid_a[0].xyz;
    float cell = grid_a[0].w;
    int3 dims = (int3)((int)grid_dims[0].x, (int)grid_dims[0].y, (int)grid_dims[0].z);
    float3 inv = 1.0f / rd;
    float3 t0s = (origin - ro) * inv;
    float3 t1s = (origin + (convert_float3(dims) * cell) - ro) * inv;
    float3 tn3 = fmin(t0s, t1s), tf3 = fmax(t0s, t1s);
    float t_entry = fmax(fmax(tn3.x, tn3.y), tn3.z);
    float t_exit = fmin(fmin(tf3.x, tf3.y), tf3.z);
    if (t_entry > t_exit || t_exit < EPS || t_entry > max_t) return -1;

    float t_start = fmax(t_entry, 0.0f);
    float3 p = ro + rd * t_start;
    int3 cell_i = convert_int3(floor((p - origin) / cell));
    cell_i = clamp(cell_i, (int3)0, dims - 1);
    int3 step = convert_int3(sign(rd));
    float3 t_delta = fabs(cell * inv);
    /* distance along the ray to the next boundary per axis */
    float3 t_next;
    if (fabs(rd.x) < 1e-9f) t_next.x = 1e30f;
    else { float b = origin.x + (convert_float3(cell_i).x + (step.x > 0 ? 1.0f : 0.0f)) * cell; t_next.x = (b - ro.x) / rd.x; }
    if (fabs(rd.y) < 1e-9f) t_next.y = 1e30f;
    else { float b = origin.y + (convert_float3(cell_i).y + (step.y > 0 ? 1.0f : 0.0f)) * cell; t_next.y = (b - ro.y) / rd.y; }
    if (fabs(rd.z) < 1e-9f) t_next.z = 1e30f;
    else { float b = origin.z + (convert_float3(cell_i).z + (step.z > 0 ? 1.0f : 0.0f)) * cell; t_next.z = (b - ro.z) / rd.z; }

    float best_t = fmin(max_t, t_exit);
    int best_tri = -1;
    for (;;) {
        /* z-major, matching the host CSR builder (z*gy*gx + y*gx + x) */
        uint ci = (uint)(cell_i.z * dims.y * dims.x + cell_i.y * dims.x + cell_i.x);
        uint off = grid_off[ci], end = grid_off[ci + 1u];
        for (; off < end; off++) {
            int tri = (int)grid_tri[off];
            float t = tri_hit(tris, tri, ro, rd);
            if (t > 0.0f && t < best_t) {
                best_t = t;
                best_tri = tri;
                if (occlusion_only) return best_tri;
            }
        }
        int axis = (t_next.x < t_next.y)
            ? ((t_next.x < t_next.z) ? 0 : 2)
            : ((t_next.y < t_next.z) ? 1 : 2);
        float tn = axis == 0 ? t_next.x : (axis == 1 ? t_next.y : t_next.z);
        if (tn > best_t) break;
        if (axis == 0) { cell_i.x += step.x; t_next.x += t_delta.x; if (cell_i.x < 0 || cell_i.x >= dims.x) break; }
        else if (axis == 1) { cell_i.y += step.y; t_next.y += t_delta.y; if (cell_i.y < 0 || cell_i.y >= dims.y) break; }
        else { cell_i.z += step.z; t_next.z += t_delta.z; if (cell_i.z < 0 || cell_i.z >= dims.z) break; }
    }
    *best_t_out = best_t;
    return best_tri;
}

/* __GRID_CONT__ */

struct Hit trace(const float3 ro, const float3 rd, const int4 counts,
                 __global const float4 *sph,
                 __global const float4 *box_min, __global const float4 *box_max,
                 __global const float4 *cyl_b, __global const float4 *cyl_h,
                 __global const float4 *plane_pos,
                 const int tri_count, __global const float *tris,
                 __global const float4 *grid_a, __global const uint4 *grid_dims,
                 __global const uint *grid_off, __global const uint *grid_tri) {
    struct Hit h;
    h.t = 1e30f; h.kind = -1; h.idx = -1; h.n = (float3)(0.0f, 1.0f, 0.0f);
    for (int i = 0; i < counts.x; i++) {
        float2 t = sph_hit(sph[i], ro, rd);
        if (t.x > 0.0f && t.x < h.t) {
            float3 p = ro + rd * t.x;
            h.t = t.x; h.n = normalize(p - sph[i].xyz); h.kind = 0; h.idx = i;
        }
    }
    for (int i = 0; i < counts.y; i++) {
        float2 t = box_hit(box_min[i], box_max[i], ro, rd);
        if (t.x > 0.0f && t.x < h.t) {
            float3 p = ro + rd * t.x;
            float3 c = 0.5f * (box_min[i].xyz + box_max[i].xyz);
            float3 d = (p - c) / fmax(box_max[i].xyz - c, (float3)(1e-6f));
            float3 ad = fabs(d);
            float3 n = (float3)(0.0f);
            if (ad.x >= ad.y && ad.x >= ad.z) n = (float3)(sign(d.x), 0.0f, 0.0f);
            else if (ad.y >= ad.z) n = (float3)(0.0f, sign(d.y), 0.0f);
            else n = (float3)(0.0f, 0.0f, sign(d.z));
            h.t = t.x; h.n = n; h.kind = 1; h.idx = i;
        }
    }
    for (int i = 0; i < counts.z; i++) {
        float2 t = cyl_hit(cyl_b[i], cyl_h[i].xy, ro, rd);
        if (t.x > 0.0f && t.x < h.t) {
            float3 p = ro + rd * t.x;
            h.t = t.x;
            if (t.y > 0.5f) {
                h.n = normalize((float3)(p.x - cyl_b[i].x, 0.0f, p.z - cyl_b[i].z));
            } else {
                h.n = (float3)(0.0f, (p.y - cyl_b[i].y > 0.0f) ? 1.0f : -1.0f, 0.0f);
            }
            h.kind = 4; h.idx = i;
        }
    }
    for (int i = 0; i < counts.w; i++) {
        float t = plane_hit(plane_pos[i].w, ro, rd);
        if (t > 0.0f && t < h.t) {
            h.t = t; h.n = (float3)(0.0f, ro.y > plane_pos[i].w ? 1.0f : -1.0f, 0.0f);
            h.kind = 2; h.idx = i;
        }
    }
    float mesh_t;
    int tri = grid_traverse(ro, rd, h.t, false, &mesh_t,
                            grid_a, grid_dims, grid_off, grid_tri, tris);
    if (tri >= 0 && mesh_t < h.t) {
        h.t = mesh_t;
        h.n = normalize(cross(tri_fetch(tris, tri, 1), tri_fetch(tris, tri, 2)));
        h.kind = 3; h.idx = tri;
    }
    return h;
}

/* occluded(): shadow / occlusion test (rt_trace equivalent) */
bool occluded(const float3 ro, const float3 rd, const float max_t, const int4 counts,
              __global const float4 *sph,
              __global const float4 *box_min, __global const float4 *box_max,
              __global const float4 *cyl_b, __global const float4 *cyl_h,
              __global const float4 *plane_pos,
              const int tri_count, __global const float *tris,
              __global const float4 *grid_a, __global const uint4 *grid_dims,
              __global const uint *grid_off, __global const uint *grid_tri) {
    for (int i = 0; i < counts.x; i++) {
        float2 t = sph_hit(sph[i], ro, rd);
        if (t.x > 0.0f && t.x < max_t) return true;
    }
    for (int i = 0; i < counts.y; i++) {
        float2 t = box_hit(box_min[i], box_max[i], ro, rd);
        if (t.x > 0.0f && t.x < max_t) return true;
    }
    for (int i = 0; i < counts.z; i++) {
        float2 t = cyl_hit(cyl_b[i], cyl_h[i].xy, ro, rd);
        if (t.x > 0.0f && t.x < max_t) return true;
    }
    for (int i = 0; i < counts.w; i++) {
        float t = plane_hit(plane_pos[i].w, ro, rd);
        if (t > 0.0f && t < max_t) return true;
    }
    float mesh_t;
    int tri = grid_traverse(ro, rd, max_t, true, &mesh_t,
                            grid_a, grid_dims, grid_off, grid_tri, tris);
    (void)mesh_t;
    /* traversal already rejects hits beyond max_t */
    return tri >= 0;
}

/* sky_color port: gradient + sun glow (8 squarings = glow^256 ~ ^200) +
 * compact sun disk (dot^9000). sun_dir/sun_col are precomputed on the host
 * from scene light/camera/light_rgb(). */
float3 sky_color(const float3 rd, const float4 sun_dir, const float4 sun_col) {
    float t = clamp(rd.y * 0.5f + 0.5f, 0.0f, 1.0f);
    float3 horizon = (float3)(0.75f, 0.82f, 0.90f);
    float3 zenith = (float3)(0.25f, 0.45f, 0.78f);
    float3 c = horizon * (1.0f - t) + zenith * t;
    float glow = fmax(dot(rd, sun_dir.xyz), 0.0f);
    float glow2 = glow * glow;
    glow2 *= glow2; glow2 *= glow2; glow2 *= glow2;
    glow2 *= glow2; glow2 *= glow2; glow2 *= glow2;
    float sun_disk = powr(fmax(dot(rd, sun_dir.xyz), 0.0f), 9000.0f);
    c += sun_col.xyz * (glow2 * 0.6f + sun_disk * 120.0f);
    return c;
}

/* sample_direct_light port: stochastic light-disk sampling, shadow ray,
 * BRDF eval and MIS weight. */
float3 sample_direct_light(const float3 p, const float3 n, const float3 view,
                           const float3 albedo, const float reflection,
                           const float shininess, uint *rng,
                           const int4 counts, const int light_count,
                           __global const float4 *lpos, __global const float4 *lcol,
                           __global const float4 *lrad,
                           __global const float4 *sph,
                           __global const float4 *box_min, __global const float4 *box_max,
                           __global const float4 *cyl_b, __global const float4 *cyl_h,
                           __global const float4 *plane_pos,
                           const int tri_count, __global const float *tris,
                           __global const float4 *grid_a, __global const uint4 *grid_dims,
                           __global const uint *grid_off, __global const uint *grid_tri) {
    if (light_count <= 0) return (float3)(0.0f);
    float3 result = (float3)(0.0f);
    for (int li = 0; li < light_count; li++) {
        float3 lp = lpos[li].xyz;
        float radius = lrad[li].x;
        if (radius > 0.0f) {
            float3 axis = normalize(lp - p);
            float3 u, v;
            tangent_basis(axis, &u, &v);
            float r = sqrt(rng_float(rng)) * radius;
            float phi = 2.0f * PI_F * rng_float(rng);
            lp += u * (r * cos(phi)) + v * (r * sin(phi));
        }
        float3 delta = lp - p;
        float distance_sq = dot(delta, delta);
        if (distance_sq <= 1e-6f) continue;
        float distance = sqrt(distance_sq);
        float3 l = delta / distance;
        float no_l = fmax(dot(n, l), 0.0f);
        if (no_l <= 0.0f) continue;
        if (occluded(p + n * 1e-3f, l, distance, counts, sph, box_min, box_max,
                     cyl_b, cyl_h, plane_pos, tri_count, tris,
                     grid_a, grid_dims, grid_off, grid_tri)) continue;
        float bsdf_pdf;
        float3 f;
        brdf_eval(albedo, n, view, l, reflection, shininess, &bsdf_pdf, &f);
        if (bsdf_pdf <= 0.0f) continue;
        float light_pdf = 1.0f;
        if (radius > 0.0f) {
            float area = PI_F * radius * radius;
            float facing = fmax(dot(-l, normalize(lpos[li].xyz - p)), 0.05f);
            light_pdf = distance_sq / (area * facing);
        }
        float mis = radius > 0.0f
            ? (light_pdf * light_pdf) / (light_pdf * light_pdf + bsdf_pdf * bsdf_pdf)
            : 1.0f;
        float3 incoming = lcol[li].xyz * (lpos[li].w / (4.0f * PI_F * distance_sq));
        result += f * incoming * (no_l * mis / fmax(light_pdf, 1e-5f));
    }
    return result;
}

/* per-object material fetch:
 *  mat: (albedo.xyz, reflection)  emi: (emission.xyz, shininess)
 *  texA: (texture_type, texture_scale, texture_strength, 0)
 *  texB: (texture_color.xyz, 0) */
void fetch_material(const int kind, const int idx,
                    __global const float4 *sph_mat, __global const float4 *sph_emi,
                    __global const float4 *sph_texA, __global const float4 *sph_texB,
                    __global const float4 *box_mat, __global const float4 *box_emi,
                    __global const float4 *box_texA, __global const float4 *box_texB,
                    __global const float4 *cyl_mat, __global const float4 *cyl_emi,
                    __global const float4 *cyl_texA, __global const float4 *cyl_texB,
                    __global const float4 *plane_mat, __global const float4 *plane_emi,
                    __global const float4 *plane_texA, __global const float4 *plane_texB,
                    const float4 bunny_mat, const float4 bunny_emi,
                    float3 *albedo_base, float3 *emission, float *reflection,
                    float *shininess, float4 *texA, float4 *texB) {
    if (kind == 0) {
        *albedo_base = sph_mat[idx].xyz; *emission = sph_emi[idx].xyz;
        *reflection = sph_mat[idx].w; *shininess = sph_emi[idx].w;
        *texA = sph_texA[idx]; *texB = sph_texB[idx];
    } else if (kind == 1) {
        *albedo_base = box_mat[idx].xyz; *emission = box_emi[idx].xyz;
        *reflection = box_mat[idx].w; *shininess = box_emi[idx].w;
        *texA = box_texA[idx]; *texB = box_texB[idx];
    } else if (kind == 2) {
        *albedo_base = plane_mat[idx].xyz; *emission = plane_emi[idx].xyz;
        *reflection = plane_mat[idx].w; *shininess = plane_emi[idx].w;
        *texA = plane_texA[idx]; *texB = plane_texB[idx];
    } else if (kind == 4) {
        *albedo_base = cyl_mat[idx].xyz; *emission = cyl_emi[idx].xyz;
        *reflection = cyl_mat[idx].w; *shininess = cyl_emi[idx].w;
        *texA = cyl_texA[idx]; *texB = cyl_texB[idx];
    } else {
        *albedo_base = bunny_mat.xyz; *emission = bunny_emi.xyz;
        *reflection = bunny_mat.w; *shininess = bunny_emi.w;
        *texA = (float4)(0.0f); *texB = (float4)(0.0f);
    }
}

/* path_radiance port: iterative MIS path tracer, max_depth 4, Russian
 * roulette from depth 2, fog support (scene_small has fog_density 0).
 * Shared per-pixel body for rt_main (full frame) and rt_tiles (hybrid
 * batches). per_sample_seed != 0 re-seeds per sample exactly like the CPU
 * tracer (pixel_seed + sample*0x9e3779b9) so a pixel's RNG stream never
 * depends on which device rendered it; rt_main keeps the legacy seed-once
 * stream. */
float3 path_pixel(const int x, const int y, const int W, const int H,
                   const int spp,
                   const int sample_base,
                  const float4 cam_pos, const float4 fwd, const float4 right,
                  const float4 up, const float2 aspect_tan, const int4 counts,
                  const int tri_count, const int light_count,
                  const float4 fog, const float4 sun_dir, const float4 sun_col,
                  __global const float4 *sph, __global const float4 *sph_mat,
                  __global const float4 *sph_emi,
                  __global const float4 *sph_texA, __global const float4 *sph_texB,
                  __global const float4 *box_min, __global const float4 *box_max,
                  __global const float4 *box_mat, __global const float4 *box_emi,
                  __global const float4 *box_texA, __global const float4 *box_texB,
                  __global const float4 *cyl_b, __global const float4 *cyl_h,
                  __global const float4 *cyl_mat, __global const float4 *cyl_emi,
                  __global const float4 *cyl_texA, __global const float4 *cyl_texB,
                  __global const float4 *plane_pos, __global const float4 *plane_mat,
                  __global const float4 *plane_emi,
                  __global const float4 *plane_texA, __global const float4 *plane_texB,
                  __global const float4 *lpos, __global const float4 *lcol,
                  __global const float4 *lrad,
                  __global const float *tris,
                  __global const float4 *grid_a, __global const uint4 *grid_dims,
                  __global const uint *grid_off, __global const uint *grid_tri,
                  const float4 bunny_mat, const float4 bunny_emi,
                  const int per_sample_seed) {
    uint rng = pixel_seed(x, y);
    float3 result = (float3)(0.0f);
    int sample_count = spp < 1 ? 1 : spp;

    for (int sample = 0; sample < sample_count; sample++) {
        if (per_sample_seed != 0) rng = pixel_seed(x, y) +
            (uint)(sample_base + sample) * 0x9e3779b9u;
        float jx = rng_float(&rng);
        float jy = rng_float(&rng);
        float u = ((float)x + jx) / (float)W * 2.0f - 1.0f;
        float v = 1.0f - ((float)y + jy) / (float)H * 2.0f;
        float3 ro = cam_pos.xyz;
        float3 rd = normalize(fwd.xyz + right.xyz * (u * aspect_tan.x)
                              + up.xyz * (v * aspect_tan.y));

        float3 throughput = (float3)(1.0f);
        for (int depth = 0; depth <= MAX_DEPTH; depth++) {
            struct Hit h = trace(ro, rd, counts, sph, box_min, box_max, cyl_b, cyl_h,
                                 plane_pos, tri_count, tris,
                                 grid_a, grid_dims, grid_off, grid_tri);
            if (h.kind < 0) {
                float3 environment = sky_color(rd, sun_dir, sun_col);
                if (fog.x > 0.0f) {
                    float transmittance = exp(-fog.x * 100.0f);
                    result += throughput * fog.yzw * (1.0f - transmittance);
                    environment *= transmittance;
                }
                result += throughput * environment;
                break;
            }
            float3 p = ro + rd * h.t;
            if (fog.x > 0.0f) {
                float transmittance = exp(-fog.x * h.t);
                result += throughput * fog.yzw * (1.0f - transmittance);
                throughput *= transmittance;
            }
            float3 albedo_base, emission, albedo;
            float reflection, shininess;
            float4 texA, texB;
            fetch_material(h.kind, h.idx, sph_mat, sph_emi, sph_texA, sph_texB,
                           box_mat, box_emi, box_texA, box_texB,
                           cyl_mat, cyl_emi, cyl_texA, cyl_texB,
                           plane_mat, plane_emi, plane_texA, plane_texB,
                           bunny_mat, bunny_emi,
                           &albedo_base, &emission, &reflection, &shininess, &texA, &texB);
            albedo = material_albedo(albedo_base, texA, texB, p);
            if (h.kind == 2 && plane_pos[h.idx].x > 0.5f && (int)texA.x == 0) {
                if ((((int)floor(p.x) + (int)floor(p.z)) & 1) != 0) albedo *= 0.35f;
            }

            result += throughput * emission;
            /* sky is an environment emitter: small cosine-weighted estimate */
            float3 environment = sky_color(h.n, sun_dir, sun_col);
            result += throughput * (albedo * environment) * (0.08f / PI_F);

            float3 view = -rd;
            result += throughput * sample_direct_light(p, h.n, view, albedo,
                reflection, shininess, &rng, counts, light_count, lpos, lcol, lrad,
                sph, box_min, box_max, cyl_b, cyl_h, plane_pos, tri_count, tris,
                grid_a, grid_dims, grid_off, grid_tri);

            if (depth == MAX_DEPTH) break;
            float bsdf_pdf;
            float3 f, next_dir;
            f = sample_bsdf(albedo, h.n, view, reflection, shininess, &rng,
                            &next_dir, &bsdf_pdf);
            if (bsdf_pdf <= 1e-6f) break;
            float no_l = fmax(dot(h.n, next_dir), 0.0f);
            if (no_l <= 0.0f) break;
            throughput *= f * (no_l / bsdf_pdf);
            if (depth >= 2) {
                float q = clamp01(fmax(lum(throughput),
                    fmax(throughput.x, fmax(throughput.y, throughput.z))));
                if (rng_float(&rng) > q) break;
                throughput *= 1.0f / fmax(q, 1e-3f);
            }
            ro = p + h.n * 1e-3f;
            rd = next_dir;
        }
    }
    return result / (float)sample_count;
}

__kernel void rt_main(__global float4 *out, __global const float *frame,
                      __global const float4 *sph, __global const float4 *sph_mat,
                      __global const float4 *sph_emi,
                      __global const float4 *sph_texA, __global const float4 *sph_texB,
                      __global const float4 *box_min, __global const float4 *box_max,
                      __global const float4 *box_mat, __global const float4 *box_emi,
                      __global const float4 *box_texA, __global const float4 *box_texB,
                      __global const float4 *cyl_b, __global const float4 *cyl_h,
                      __global const float4 *cyl_mat, __global const float4 *cyl_emi,
                      __global const float4 *cyl_texA, __global const float4 *cyl_texB,
                      __global const float4 *plane_pos, __global const float4 *plane_mat,
                      __global const float4 *plane_emi,
                      __global const float4 *plane_texA, __global const float4 *plane_texB,
                      __global const float4 *lpos, __global const float4 *lcol,
                      __global const float4 *lrad,
                      __global const float *tris,
                      __global const float4 *grid_a, __global const uint4 *grid_dims,
                      __global const uint *grid_off, __global const uint *grid_tri) {
    const int W = (int)frame[F_W], H = (int)frame[F_H];
    const int x = (int)get_global_id(0);
    const int y = (int)get_global_id(1);
    if (x >= W || y >= H) return;

    const float4 cam_pos = vload4(0, frame + F_CAM);
    const float4 fwd = vload4(0, frame + F_FWD);
    const float4 right = vload4(0, frame + F_RIGHT);
    const float4 up = vload4(0, frame + F_UP);
    const float2 aspect_tan = (float2)(frame[F_ASPECT], frame[F_ASPECT + 1]);
    const int4 counts = (int4)((int)frame[F_COUNTS], (int)frame[F_COUNTS + 1],
                               (int)frame[F_COUNTS + 2], (int)frame[F_COUNTS + 3]);
    const int tri_count = (int)frame[F_TRI];
    const int light_count = (int)frame[F_LIGHTS];
    const float4 fog = vload4(0, frame + F_FOG);
    const float4 sun_dir = vload4(0, frame + F_SUN_DIR);
    const float4 sun_col = vload4(0, frame + F_SUN_COL);
    const float4 bunny_mat = vload4(0, frame + F_BMAT);
    const float4 bunny_emi = vload4(0, frame + F_BEMI);
     const int spp = (int)frame[F_SPP];
     const int sample_base = (int)frame[F_SAMPLE_BASE];
     const int accumulating = (int)frame[F_ACCUM];

     float3 result = path_pixel(x, y, W, H, spp, sample_base,
                               cam_pos, fwd, right, up, aspect_tan, counts,
                               tri_count, light_count, fog, sun_dir, sun_col,
                               sph, sph_mat, sph_emi, sph_texA, sph_texB,
                               box_min, box_max, box_mat, box_emi, box_texA, box_texB,
                               cyl_b, cyl_h, cyl_mat, cyl_emi, cyl_texA, cyl_texB,
                               plane_pos, plane_mat, plane_emi, plane_texA, plane_texB,
                               lpos, lcol, lrad, tris,
                               grid_a, grid_dims, grid_off, grid_tri,
                                bunny_mat, bunny_emi, accumulating ? 1 : 0);

     const int o = y * W + x;
     if (accumulating) {
         const float old_count = (float)sample_base;
         const float new_count = old_count + (float)(spp < 1 ? 1 : spp);
         float3 old = out[o].xyz;
         result = old_count > 0.0f
             ? (old * old_count + result * (float)spp) / new_count
             : result;
     }
     out[o] = (float4)(result, 1.0f);
}

/* Hybrid tile worker: one workgroup per tile (tile*tile work-items),
 * tile_batch holds (x0, y0) pairs in Morton order. Writes float4 HDR for
 * its pixels into the same film layout as rt_main; the host composites
 * CPU and GPU tiles and runs the shared post chain. Uses the CPU RNG
 * scheme (per_sample_seed = 1). */
__kernel void rt_tiles(__global float4 *out, __global const float *frame,
                       __global const float4 *sph, __global const float4 *sph_mat,
                       __global const float4 *sph_emi,
                       __global const float4 *sph_texA, __global const float4 *sph_texB,
                       __global const float4 *box_min, __global const float4 *box_max,
                       __global const float4 *box_mat, __global const float4 *box_emi,
                       __global const float4 *box_texA, __global const float4 *box_texB,
                       __global const float4 *cyl_b, __global const float4 *cyl_h,
                       __global const float4 *cyl_mat, __global const float4 *cyl_emi,
                       __global const float4 *cyl_texA, __global const float4 *cyl_texB,
                       __global const float4 *plane_pos, __global const float4 *plane_mat,
                       __global const float4 *plane_emi,
                       __global const float4 *plane_texA, __global const float4 *plane_texB,
                       __global const float4 *lpos, __global const float4 *lcol,
                       __global const float4 *lrad,
                       __global const float *tris,
                       __global const float4 *grid_a, __global const uint4 *grid_dims,
                       __global const uint *grid_off, __global const uint *grid_tri,
                       __global const int *tile_batch, const int tile) {
    const int lid = (int)get_local_id(0);
    const int tile_id = (int)get_group_id(0);
    const int W = (int)frame[F_W], H = (int)frame[F_H];
    const int x = tile_batch[tile_id * 2] + lid % tile;
    const int y = tile_batch[tile_id * 2 + 1] + lid / tile;
    if (x >= W || y >= H) return;

    const float4 cam_pos = vload4(0, frame + F_CAM);
    const float4 fwd = vload4(0, frame + F_FWD);
    const float4 right = vload4(0, frame + F_RIGHT);
    const float4 up = vload4(0, frame + F_UP);
    const float2 aspect_tan = (float2)(frame[F_ASPECT], frame[F_ASPECT + 1]);
    const int4 counts = (int4)((int)frame[F_COUNTS], (int)frame[F_COUNTS + 1],
                               (int)frame[F_COUNTS + 2], (int)frame[F_COUNTS + 3]);
    const int tri_count = (int)frame[F_TRI];
    const int light_count = (int)frame[F_LIGHTS];
    const float4 fog = vload4(0, frame + F_FOG);
    const float4 sun_dir = vload4(0, frame + F_SUN_DIR);
    const float4 sun_col = vload4(0, frame + F_SUN_COL);
    const float4 bunny_mat = vload4(0, frame + F_BMAT);
    const float4 bunny_emi = vload4(0, frame + F_BEMI);
    const int spp = (int)frame[F_SPP];

     float3 result = path_pixel(x, y, W, H, spp, 0,
                               cam_pos, fwd, right, up, aspect_tan, counts,
                               tri_count, light_count, fog, sun_dir, sun_col,
                               sph, sph_mat, sph_emi, sph_texA, sph_texB,
                               box_min, box_max, box_mat, box_emi, box_texA, box_texB,
                               cyl_b, cyl_h, cyl_mat, cyl_emi, cyl_texA, cyl_texB,
                               plane_pos, plane_mat, plane_emi, plane_texA, plane_texB,
                               lpos, lcol, lrad, tris,
                               grid_a, grid_dims, grid_off, grid_tri,
                               bunny_mat, bunny_emi, 1);
    out[y * W + x] = (float4)(result, 1.0f);
}

/* ------------------------------------------------------------------ post
 * GPU port of the CPU post chain (postprocess_hdr / exposed_hdr), kept
 * mathematically identical including the bloom's per-ring nested division.
 * Adaptation state stays on the host via rt_exposure_from_logavg(). */

/* Sum of log(delta + L) over all pixels; one partial per workgroup. Host
 * sums the partials as doubles and calls rt_exposure_from_logavg(). */
__kernel void rt_logavg(__global const float4 *hdr, const int npix,
                        __local float *lsum, __global float *partials) {
    const int gid = (int)get_global_id(0);
    const int gsz = (int)get_global_size(0);
    const int lid = (int)get_local_id(0);
    float acc = 0.0f;
    for (int i = gid; i < npix; i += gsz) {
        float l = fmax(lum(hdr[i].xyz), 0.0f);
        acc += log(1e-4f + l);
    }
    lsum[lid] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = (int)get_local_size(0) >> 1; s > 0; s >>= 1) {
        if (lid < s) lsum[lid] += lsum[lid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) partials[get_group_id(0)] = lsum[0];
}

__constant int   BLOOM_OFFSETS[4] = { 1, 2, 4, 8 };
__constant float BLOOM_WEIGHTS[4] = { 0.20f, 0.10f, 0.055f, 0.025f };

uchar srgb_byte(float x) {
    if (!(x > 0.0f)) return 0;
    x = fmax(x, 0.0f);
    float e = x <= 0.0031308f
        ? 12.92f * x
        : 1.055f * powr(x, 1.0f / 2.4f) - 0.055f;
    if (e > 1.0f) e = 1.0f;
    return convert_uchar_sat(e * 255.0f + 0.5f);
}

/* Tone map the internal HDR film to 8-bit sRGB (rgba). */
__kernel void rt_post(__global const float4 *hdr, __global uchar4 *rgb8,
                      const int W, const int H, const float exposure) {
    const int x = (int)get_global_id(0);
    const int y = (int)get_global_id(1);
    if (x >= W || y >= H) return;

    float3 bloom = (float3)(0.0f);
    for (int i = 0; i < 4; i++) {
        const int distance = BLOOM_OFFSETS[i];
        int samples = 0;
        const int px[4] = { x - distance, x + distance, x, x };
        const int py[4] = { y, y, y - distance, y + distance };
        for (int k = 0; k < 4; k++) {
            if (px[k] < 0 || px[k] >= W || py[k] < 0 || py[k] >= H) continue;
            float3 s = hdr[py[k] * W + px[k]].xyz * exposure;
            if (lum(s) <= 1.0f) continue;
            bloom += s * BLOOM_WEIGHTS[i];
            samples++;
        }
        if (samples > 0) bloom /= (float)samples;
    }
    float3 c = hdr[y * W + x].xyz * exposure + bloom;
    c = c / (1.0f + c);
    rgb8[y * W + x] = (uchar4)(srgb_byte(c.x), srgb_byte(c.y), srgb_byte(c.z), 255);
}

/* ---------------------------------------------------------------- upscale
 * Spatial upscale of the tone-mapped film: Catmull-Rom bicubic, edge-clamped
 * sampling, then optional CAS sharpen as a separate pass. */

float4 cr_weights(const float t) {
    const float t2 = t * t;
    const float t3 = t2 * t;
    return (float4)(-0.5f * t3 + t2 - 0.5f * t,
                     1.5f * t3 - 2.5f * t2 + 1.0f,
                    -1.5f * t3 + 2.0f * t2 + 0.5f * t,
                     0.5f * t3 - 0.5f * t2);
}

float3 cr_fetch(__global const uchar4 *src, const int IW, const int IH,
                int ix, int iy) {
    ix = clamp(ix, 0, IW - 1);
    iy = clamp(iy, 0, IH - 1);
    uchar4 s = src[iy * IW + ix];
    return convert_float3(s.xyz);
}

float3 cr_sample(__global const uchar4 *src, const int IW, const int IH,
                 const float fx, const float fy) {
    const float flx = floor(fx), fly = floor(fy);
    const float4 wx = cr_weights(fx - flx);
    const float4 wy = cr_weights(fy - fly);
    const int x0 = (int)flx - 1;
    const int y0 = (int)fly - 1;
    float3 acc = (float3)(0.0f);
    for (int j = 0; j < 4; j++) {
        float3 row = (float3)(0.0f);
        for (int i = 0; i < 4; i++) {
            row += cr_fetch(src, IW, IH, x0 + i, y0 + j) * wx[i];
        }
        acc += row * wy[j];
    }
    return acc;
}

__kernel void rt_upscale(__global const uchar4 *src, __global uchar4 *dst,
                         const int IW, const int IH, const int W, const int H) {
    const int x = (int)get_global_id(0);
    const int y = (int)get_global_id(1);
    if (x >= W || y >= H) return;
    const float fx = ((float)x + 0.5f) * ((float)IW / (float)W) - 0.5f;
    const float fy = ((float)y + 0.5f) * ((float)IH / (float)H) - 0.5f;
    float3 c = cr_sample(src, IW, IH, fx, fy);
    c = clamp(c, (float3)(0.0f), (float3)(255.0f));
    dst[y * W + x] = (uchar4)(convert_uchar_sat(c.x + 0.5f),
                              convert_uchar_sat(c.y + 0.5f),
                              convert_uchar_sat(c.z + 0.5f), 255);
}

/* CAS-style sharpen with halo suppression (clamped to the 5-tap range). */
__kernel void rt_cas(__global const uchar4 *src, __global uchar4 *dst,
                     const int W, const int H, const float amount) {
    const int x = (int)get_global_id(0);
    const int y = (int)get_global_id(1);
    if (x >= W || y >= H) return;
    const int xm = max(x - 1, 0), xp = min(x + 1, W - 1);
    const int ym = max(y - 1, 0), yp = min(y + 1, H - 1);
    float3 c  = convert_float3(src[y * W + x].xyz);
    float3 n1 = convert_float3(src[y * W + xm].xyz);
    float3 n2 = convert_float3(src[y * W + xp].xyz);
    float3 n3 = convert_float3(src[ym * W + x].xyz);
    float3 n4 = convert_float3(src[yp * W + x].xyz);
    float3 blur = (n1 + n2 + n3 + n4) * 0.25f;
    float3 s = c + (c - blur) * amount;
    float3 lo = fmin(c, fmin(n1, fmin(n2, fmin(n3, n4))));
    float3 hi = fmax(c, fmax(n1, fmax(n2, fmax(n3, n4))));
    s = clamp(s, lo, hi);
    dst[y * W + x] = (uchar4)(convert_uchar_sat(s.x + 0.5f),
                              convert_uchar_sat(s.y + 0.5f),
                              convert_uchar_sat(s.z + 0.5f), 255);
}

/* Zero-copy present (cl_khr_gl_sharing): copy the final RGBA8 frame into the
 * GL-shared texture so the viewer draws it without a readback round-trip. */
__kernel void rt_present(__global const uchar4 *src, __write_only image2d_t dst,
                         const int W, const int H) {
    const int x = (int)get_global_id(0);
    const int y = (int)get_global_id(1);
    if (x >= W || y >= H) return;
     const uchar4 c = src[y * W + x];
     /* GL_RGBA8 is a normalized image: integer writes produce black on the
      * Intel sharing path, while normalized float writes preserve RGBA8. */
     write_imagef(dst, (int2)(x, y),
                  (float4)((float)c.x / 255.0f, (float)c.y / 255.0f,
                           (float)c.z / 255.0f, 1.0f));
}
