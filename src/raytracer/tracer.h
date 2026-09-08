#ifndef TRACER_H
#define TRACER_H

#include <math.h>

typedef struct { float x, y, z; } V3;
static inline V3 v3(float x, float y, float z) { return (V3){x, y, z}; }
static inline V3 vadd(V3 a, V3 b) { return v3(a.x+b.x, a.y+b.y, a.z+b.z); }
static inline V3 vsub(V3 a, V3 b) { return v3(a.x-b.x, a.y-b.y, a.z-b.z); }
static inline V3 vscale(V3 a, float s) { return v3(a.x*s, a.y*s, a.z*s); }
static inline V3 vmul(V3 a, V3 b) { return v3(a.x*b.x, a.y*b.y, a.z*b.z); }
static inline float vdot(V3 a, V3 b) { return a.x*b.x+a.y*b.y+a.z*b.z; }
static inline V3 vcross(V3 a, V3 b) {
    return v3(a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x);
}
static inline V3 vnorm(V3 a) {
    float len = sqrtf(vdot(a, a));
    return len > 0.0f ? vscale(a, 1.0f/len) : v3(0, 0, 0);
}
static inline float vlen(V3 a) { return sqrtf(vdot(a, a)); }

typedef struct {
    V3 albedo;
    float reflection;
    float shininess;
    V3 emission;
    int texture;
    float texture_scale;
    float texture_strength;
    V3 texture_color;
} Material;
typedef enum {
    RT_TEX_NONE = 0,
    RT_TEX_CHECKER,
    RT_TEX_GRASS,
    RT_TEX_DIRT,
    RT_TEX_STONE,
    RT_TEX_WALL,
    RT_TEX_ROOF
} RtTexture;
#define RT_MAX_MESH_VERTICES 40000
#define RT_MAX_MESH_TRIANGLES 70000
#define RT_MAX_MESH_NODES 140000
#define RT_MAX_MESHES 8
typedef struct {
    V3 min, max;
    int left, right;
    int start, count;
} RtMeshNode;
typedef struct {
    V3 vertices[RT_MAX_MESH_VERTICES];
    int triangles[RT_MAX_MESH_TRIANGLES][3];
    int vertex_count, triangle_count;
    int triangle_indices[RT_MAX_MESH_TRIANGLES];
    int node_count;
    RtMeshNode nodes[RT_MAX_MESH_NODES];
} RtMesh;
typedef enum { RT_BOX, RT_SPHERE, RT_CYLINDER, RT_PLANE, RT_MESH } Shape;
typedef struct {
    Shape shape;
    Material material;
    union {
        struct { V3 min, max; } box;
        struct { V3 center; float radius; } sphere;
        /* Vertical, closed cylinder: base is the center of its bottom cap. */
        struct { V3 base; float radius, height; } cylinder;
        struct { float y; int checker; } plane;
        struct { int mesh; V3 min, max; } mesh;
    } geometry;
} Object;

#define RT_MAX_OBJECTS 192
#define RT_MAX_LIGHTS 8
#define RT_EPSILON 0.0005f
/* Binned SAH build constants shared by the object and mesh BVH builders. */
#define RT_SAH_BINS 12
#define RT_SAH_TRAV_COST 1.0f
#define RT_SAH_PRIM_COST 1.5f
#define RT_SAH_MAX_DEPTH 32
typedef struct {
    V3 position;
    V3 color;
    float power;
    float radius;
    int cast_shadows;
    float temperature;
} RtPointLight;
typedef struct {
    V3 min, max;
    int left, right;
    int start, count;
} RtAccelNode;
typedef struct {
    Object objects[RT_MAX_OBJECTS];
    int count;
    int accel_count;
    int accel_node_count;
    int accel_indices[RT_MAX_OBJECTS];
    RtAccelNode accel_nodes[RT_MAX_OBJECTS * 2];
    RtPointLight lights[RT_MAX_LIGHTS];
    RtMesh meshes[RT_MAX_MESHES];
    int mesh_count;
    int mesh_enabled;
    int render_meshes;
    int light_count;
    int red_ball_object;
    int blue_ball_object;
    V3 camera, target;
    float fov;
    V3 light, light_color;
    float light_power, light_radius;
    float fog_density;
    V3 fog_color;
} Scene;
typedef struct { float t; V3 normal; int object; } Hit;

/* Directions must be normalized; t_max limits both nearest-hit and shadow rays. */
int rt_intersect(const Object *object, V3 origin, V3 direction, float t_max, Hit *hit);
int rt_trace(const Scene *scene, V3 origin, V3 direction, float t_max, Hit *hit);
void rt_build_accel(Scene *scene);
int rt_occluded(const Scene *scene, V3 origin, V3 light);
V3 rt_radiance(const Scene *scene, V3 origin, V3 direction, int depth);
/* RGB, top-down. samples is 1, 4, or 16. Returns zero for invalid input/nonfinite pixels. */
int rt_render(const Scene *scene, unsigned char *rgb, int width, int height, int samples);
/* Render only [y0,y1), allowing independent rows to run on worker threads. */
int rt_render_rows(const Scene *scene, unsigned char *rgb, int width, int height,
                  int samples, int y0, int y1, int shadow_samples, int max_depth);
/* Fast interactive path: traces one ray per block and fills the full buffer. */
int rt_render_blocked(const Scene *scene, unsigned char *rgb, int width, int height,
                     int block_size, int shadow_samples, int max_depth);
/* True full-resolution path: exactly one primary ray per output pixel. */
int rt_render_exact(const Scene *scene, unsigned char *rgb, int width, int height,
                    int shadow_samples, int max_depth);
int rt_render_exact_fast(const Scene *scene, unsigned char *rgb, int width, int height,
                         int shadow_samples, int max_depth);
int rt_render_native(const Scene *scene, unsigned char *rgb, int width, int height,
                     int spp, int max_depth);
/* Trace only the listed tile origins (tx, ty pairs, tile size 16) into an
 * HDR film of width x height pixels; worker threads pop tiles dynamically. */
int rt_render_tiles_hdr(const Scene *scene, V3 *hdr, int width, int height,
                        int spp, int max_depth, const int *tiles_xy, int tile_count);
/* Morton-ordered tile-origin list ((tx, ty) pairs, tile px each) covering
 * the film; out_xy == NULL returns the tile count, else the count written
 * (or -1 if cap is too small). Shared by the CPU and hybrid schedulers. */
int rt_build_tile_origins(int width, int height, int tile, int *out_xy, int cap);
int rt_render_adaptive(const Scene *scene, unsigned char *rgb, int width, int height,
                       int spp_min, int spp_max, float epsilon, float contrast,
                       int max_depth);
int rt_render_adaptive_fast(const Scene *scene, unsigned char *rgb, int width, int height,
                            int spp_min, int spp_max);
/* Shared post chain (exposure, bloom, Reinhard, sRGB) applied to linear HDR
 * frames regardless of whether the CPU or the GPU produced them. */
int rt_postprocess_hdr(const V3 *hdr, unsigned char *rgb, int width, int height);
/* Pupil + adaptation + exposure from a precomputed SUM of log(delta + L) over
 * all pixels; updates the shared temporal adaptation state. Used by the CPU
 * post chain and by the GPU post kernels (which compute log_sum on device). */
float rt_exposure_from_logavg(double log_sum, int pixel_count);
/* Canonical light color: blackbody(temperature) when set, light->color otherwise. */
V3 rt_light_rgb(const RtPointLight *light);
void scene_block(Scene *scene);
void scene_world(Scene *scene);
void scene_small(Scene *scene);
int rt_load_mesh(Scene *scene, const char *path, V3 position, float scale, Material material);
void rt_set_frame_delta(float delta_seconds);

#endif
