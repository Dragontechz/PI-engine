/* Interactive full-resolution ray-traced world viewer.
 *
 * This is intentionally separate from ui.c and main3d.c. The scene is built
 * by tracer_scene.c, while this file owns only camera movement and presentation.
 * Adaptive mode is full-resolution by default. The optional --adaptive-res
 * flag enables the coarse render/reconstruction preview path.
 *
 * Controls:
 *   WASD       fly forward/back/left/right
 *   Q/E        move down/up
 *   Shift      sprint
 *   Mouse      look around
 *   R          reset camera
 *   F1         toggle help overlay
 *   F2         save world_viewer.png
 *   Escape     release mouse / close window
 */
#include "raylib.h"
#include "pitsr.h"
#define Material RtTracerMaterial
#include "gpu_renderer.h"
#include "tracer.h"
#undef Material
#ifdef _OPENMP
#include <omp.h>
#endif
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define WINDOW_W 1280
#define WINDOW_H 720
#define MAX_RENDER_W 1280
#define MAX_RENDER_H 720
#define TARGET_FPS 45.0
#define RENDER_BUDGET_MS 18.0
#define MOVE_SPEED 7.5f
#define SPRINT_MULTIPLIER 2.5f
#define MOUSE_SENSITIVITY 0.0025f
#define PI 3.14159265358979323846f

typedef struct {
    V3 position;
    float yaw;
    float pitch;
} ViewerCamera;

static V3 v3_add(V3 a, V3 b) { return vadd(a, b); }
static V3 v3_sub(V3 a, V3 b) { return vsub(a, b); }

static float clampf_(float x, float lo, float hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

static int render_parallel(const Scene *scene, unsigned char *rgb, int width, int height,
                           int adaptive, int adaptive_res, int spp_min, int spp_max) {
    if (!scene || !rgb || width <= 0 || height <= 0) return 0;
#ifdef _OPENMP
    omp_set_num_threads(4);
#endif
    if (!adaptive) return rt_render_native(scene, rgb, width, height, spp_max, 4);
    if (!adaptive_res) {
        return rt_render_adaptive(scene, rgb, width, height, spp_min, spp_max, 0.05f, 0.20f, 4);
    }
    /* Spatial upscale per instructions.txt: trace a smaller film, reconstruct
     * to output res. Temporal PiTSR is parked for now. */
    return rt_render_adaptive_fast(scene, rgb, width, height, spp_min, spp_max);
}

static V3 forward_vector(const ViewerCamera *camera) {
    float cp = cosf(camera->pitch);
    return vnorm(v3(cosf(camera->yaw) * cp,
                    sinf(camera->pitch),
                    sinf(camera->yaw) * cp));
}

static void reset_camera(ViewerCamera *camera) {
    /* Start in the center plaza, looking across the well and light balls. */
    camera->position = v3(0.0f, 3.4f, 7.5f);
    camera->yaw = -1.5708f;
    camera->pitch = -0.16f;
}

static void update_camera(ViewerCamera *camera, float dt) {
    V3 forward = forward_vector(camera);
    V3 flat_forward = vnorm(v3(forward.x, 0.0f, forward.z));
    V3 right = vnorm(vcross(flat_forward, v3(0.0f, 1.0f, 0.0f)));
    V3 move = v3(0.0f, 0.0f, 0.0f);
    if (IsKeyDown(KEY_W)) move = v3_add(move, flat_forward);
    if (IsKeyDown(KEY_S)) move = v3_sub(move, flat_forward);
    if (IsKeyDown(KEY_D)) move = v3_add(move, right);
    if (IsKeyDown(KEY_A)) move = v3_sub(move, right);
    if (IsKeyDown(KEY_E)) move.y += 1.0f;
    if (IsKeyDown(KEY_Q)) move.y -= 1.0f;
    float length = vlen(move);
    if (length > 0.0f) {
        float speed = MOVE_SPEED * (IsKeyDown(KEY_LEFT_SHIFT) ? SPRINT_MULTIPLIER : 1.0f);
        camera->position = v3_add(camera->position, vscale(move, speed * dt / length));
    }

    Vector2 mouse = GetMouseDelta();
    camera->yaw += mouse.x * MOUSE_SENSITIVITY;
    camera->pitch -= mouse.y * MOUSE_SENSITIVITY;
    camera->pitch = clampf_(camera->pitch, -1.45f, 1.45f);

    /* Keep exploration inside the authored diorama instead of flying forever. */
    camera->position.x = clampf_(camera->position.x, -13.0f, 13.0f);
    camera->position.y = clampf_(camera->position.y, 1.0f, 16.0f);
    camera->position.z = clampf_(camera->position.z, -13.0f, 16.0f);
}

static void update_scene_camera(Scene *scene, const ViewerCamera *camera) {
    scene->camera = camera->position;
    scene->target = v3_add(camera->position, vscale(forward_vector(camera), 10.0f));
    scene->fov = 72.0f;
}

static void push_ball(Scene *scene, int object_index, V3 camera_delta,
                      V3 *camera_position) {
    if (object_index < 0 || object_index >= scene->count) return;
    Object *ball = &scene->objects[object_index];
    V3 center = ball->geometry.sphere.center;
    V3 gap = vsub(center, *camera_position);
    gap.y = 0.0f;
    float distance = vlen(gap);
    if (distance > 1.15f || distance < 1e-5f) return;
    V3 push = v3(camera_delta.x, 0.0f, camera_delta.z);
    float push_length = vlen(push);
    if (push_length < 1e-5f) return;
    center = vadd(center, vscale(push, 1.35f));
    center.x = clampf_(center.x, -8.0f, 8.0f);
    center.z = clampf_(center.z, -7.0f, 7.0f);
    center.y = 1.15f;
    ball->geometry.sphere.center = center;
}

static int update_pushable_balls(Scene *scene, V3 camera_delta, V3 camera_position) {
    V3 before_red = scene->objects[scene->red_ball_object].geometry.sphere.center;
    V3 before_blue = scene->objects[scene->blue_ball_object].geometry.sphere.center;
    push_ball(scene, scene->red_ball_object, camera_delta, &camera_position);
    push_ball(scene, scene->blue_ball_object, camera_delta, &camera_position);
    V3 red = scene->objects[scene->red_ball_object].geometry.sphere.center;
    V3 blue = scene->objects[scene->blue_ball_object].geometry.sphere.center;
    if (scene->light_count >= 3) {
        scene->lights[1].position = red;
        scene->lights[2].position = blue;
    }
    return vlen(vsub(red, before_red)) > 1e-6f || vlen(vsub(blue, before_blue)) > 1e-6f;
}

static void present_rgb(Texture2D texture, const unsigned char *rgb, unsigned char *rgba,
                        int render_w, int render_h) {
    for (int y = 0; y < MAX_RENDER_H; y++) {
        int source_y = y * render_h / MAX_RENDER_H;
        for (int x = 0; x < MAX_RENDER_W; x++) {
            int source_x = x * render_w / MAX_RENDER_W;
            int source_i = (source_y * render_w + source_x) * 3;
            int output_i = (y * MAX_RENDER_W + x) * 4;
            rgba[output_i + 0] = rgb[source_i + 0];
            rgba[output_i + 1] = rgb[source_i + 1];
            rgba[output_i + 2] = rgb[source_i + 2];
            rgba[output_i + 3] = 255;
        }
    }
    UpdateTexture(texture, rgba);
}

static void draw_hud(double render_ms, int show_help, int gpu_active) {
    DrawRectangle(0, 0, WINDOW_W, 46, (Color){10, 13, 18, 225});
    DrawText(TextFormat("MINI WORLD / %s RAY TRACE", gpu_active ? "GPU" : "CPU"),
             20, 14, 18, (Color){235, 238, 245, 255});
    DrawText(TextFormat("%.1f ms  |  %.0f FPS", render_ms, GetFPS()), 350, 15, 16, (Color){145, 197, 156, 255});
        DrawText("WASD move   Q/E altitude   SHIFT sprint   F1 help   F2 screenshot   R reset", 670, 16, 14, (Color){170, 178, 193, 255});

    if (show_help) {
        DrawRectangle(28, 72, 390, 218, (Color){16, 20, 28, 238});
        DrawRectangleLines(28, 72, 390, 218, (Color){93, 111, 137, 255});
        DrawText("FREE-ROAM CONTROLS", 48, 92, 18, (Color){240, 177, 85, 255});
        DrawText("W A S D   move across the island", 48, 126, 15, RAYWHITE);
        DrawText("Q / E       descend / ascend", 48, 150, 15, RAYWHITE);
        DrawText("SHIFT       sprint", 48, 174, 15, RAYWHITE);
        DrawText("Mouse       look", 48, 198, 15, RAYWHITE);
        DrawText("R           reset camera", 48, 222, 15, RAYWHITE);
        DrawText("Walk into red/blue balls to push them", 48, 246, 15, (Color){235, 120, 120, 255});
        DrawText("Mirrors are placed around the plaza", 48, 270, 15, (Color){190, 215, 235, 255});
    }
}

int main(int argc, char **argv) {
    setvbuf(stderr, NULL, _IONBF, 0); /* survive hard crashes during diagnosis */
    fprintf(stderr, "viewer: main entered\n");
    static Scene scene;
    ViewerCamera camera;
    int adaptive = 1;
    int adaptive_res = 0;
    int use_gpu = 1;
    int spp_min = 4;
    int spp_max = 16;
    float scale = 0.67f;    /* upscale mode: internal = output * scale */
    float cas = 0.4f;       /* upscale mode: CAS sharpen after upscale */
    int uniform_spp = 0;    /* upscale/native mode: explicit --spp */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--mode") == 0 && i + 1 < argc) {
            const char *mode = argv[++i];
            adaptive = strcmp(mode, "native") != 0;
            adaptive_res = strcmp(mode, "upscale") == 0;
        }
        else if (strcmp(argv[i], "--adaptive-res") == 0) adaptive_res = 1;
        else if (strcmp(argv[i], "--gpu") == 0 && i + 1 < argc) use_gpu = strcmp(argv[++i], "off") != 0;
        else if (strcmp(argv[i], "--spp-min") == 0 && i + 1 < argc) spp_min = atoi(argv[++i]);
        else if (strcmp(argv[i], "--spp-max") == 0 && i + 1 < argc) spp_max = atoi(argv[++i]);
        else if (strcmp(argv[i], "--spp") == 0 && i + 1 < argc) { uniform_spp = atoi(argv[++i]); }
        else if (strcmp(argv[i], "--scale") == 0 && i + 1 < argc) scale = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--cas") == 0 && i + 1 < argc) cas = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--upscaler") == 0 && i + 1 < argc) {
            if (strcmp(argv[++i], "none") == 0) scale = 1.0f;
        }
        else if (strcmp(argv[i], "--jitter") == 0 && i + 1 < argc) {
            if (strcmp(argv[++i], "on") == 0)
                TraceLog(LOG_WARNING, "--jitter on not supported this pass; running unjittered");
        }
    }
    if (spp_min < 1) spp_min = 1;
    if (spp_max < spp_min) spp_max = spp_min;
    /* Explicit --spp is uniform: it must reach the CPU paths too, not just
     * frame_spp for the GPU (native renders with spp_max, adaptive with
     * spp_min..spp_max; pinning both makes it truly uniform). */
    if (uniform_spp > 0) {
        spp_min = uniform_spp;
        spp_max = uniform_spp;
    }
    if (scale < 0.25f) scale = 0.25f;
    if (scale > 1.0f) scale = 1.0f;
    if (cas < 0.0f) cas = 0.0f;
    if (cas > 1.0f) cas = 1.0f;
    /* GPU trace params: upscale mode traces a smaller film (instructions.txt);
     * native/adaptive trace 1:1 with the historical spp defaults. */
    int frame_spp = uniform_spp > 0 ? uniform_spp : (adaptive_res ? 1 : 2);
    int internal_w = adaptive_res ? (int)((MAX_RENDER_W * scale) + 0.5f) : MAX_RENDER_W;
    int internal_h = adaptive_res ? (int)((MAX_RENDER_H * scale) + 0.5f) : MAX_RENDER_H;
    if (internal_w < 1) internal_w = 1;
    if (internal_h < 1) internal_h = 1;
    if (scale >= 1.0f) { internal_w = MAX_RENDER_W; internal_h = MAX_RENDER_H; }
    unsigned char *rgb = (unsigned char *)malloc((size_t)MAX_RENDER_W * MAX_RENDER_H * 3);
    if (!rgb) return 1;
    unsigned char *rgba = (unsigned char *)malloc((size_t)MAX_RENDER_W * MAX_RENDER_H * 4);
    if (!rgba) {
        free(rgb);
        return 1;
    }

    scene_small(&scene);
    reset_camera(&camera);
    update_scene_camera(&scene, &camera);

    SetConfigFlags(FLAG_WINDOW_RESIZABLE);
    InitWindow(WINDOW_W, WINDOW_H, "Mini World - CPU Ray Tracer");
    SetWindowMinSize(800, 500);
    SetTargetFPS((int)TARGET_FPS);
    DisableCursor();

    unsigned char *texture_data = (unsigned char *)calloc((size_t)MAX_RENDER_W * MAX_RENDER_H * 4, 1);
    if (!texture_data) {
        CloseWindow();
        free(rgba);
        free(rgb);
        return 1;
    }
    Image image = {
        .data = texture_data,
        .width = MAX_RENDER_W,
        .height = MAX_RENDER_H,
        .mipmaps = 1,
        .format = PIXELFORMAT_UNCOMPRESSED_R8G8B8A8
    };
    Texture2D texture = LoadTextureFromImage(image);
    free(texture_data);
    SetTextureFilter(texture, TEXTURE_FILTER_POINT);
    GpuRenderer *gpu = use_gpu ? GpuRenderer_Create(MAX_RENDER_W, MAX_RENDER_H) : NULL;
    int show_help = 1;
    int startup_screenshot_saved = 0;
    double render_ms = 0.0;
    int frame_index = 0;

    while (!WindowShouldClose()) {
        float dt = GetFrameTime();
        if (dt > 0.1f) dt = 0.1f;
        if (IsKeyPressed(KEY_R)) reset_camera(&camera);
        if (IsKeyPressed(KEY_F1)) show_help = !show_help;
        if (IsKeyPressed(KEY_ESCAPE)) {
            if (IsCursorHidden()) EnableCursor();
            else DisableCursor();
        }
        V3 previous_camera = camera.position;
        if (IsCursorHidden()) update_camera(&camera, dt);
        V3 camera_delta = vsub(camera.position, previous_camera);
        int balls_moved = update_pushable_balls(&scene, camera_delta, camera.position);
        if (balls_moved) rt_build_accel(&scene);
        update_scene_camera(&scene, &camera);

        double start = GetTime();
        rt_set_frame_delta(dt);
        int frame_w = MAX_RENDER_W;
        int frame_h = MAX_RENDER_H;
        int gpu_used = gpu && GpuRenderer_Render(gpu, &scene, rgb, frame_w, frame_h,
                                                 internal_w, internal_h, frame_spp, cas);
        if (!gpu_used && !render_parallel(&scene, rgb, frame_w, frame_h, adaptive, adaptive_res,
                                          spp_min, spp_max)) {
            TraceLog(LOG_ERROR, "Ray-traced frame failed");
            break;
        }
        (void)frame_index++;
        render_ms = (GetTime() - start) * 1000.0;
        if (frame_index % 30 == 0 || frame_index <= 3) {
            TraceLog(LOG_INFO, "frame %d: %.1f ms (%.1f fps)", frame_index, render_ms,
                     render_ms > 0.0 ? 1000.0 / render_ms : 0.0);
        }
        present_rgb(texture, rgb, rgba, frame_w, frame_h);

        BeginDrawing();
        ClearBackground((Color){8, 10, 14, 255});
        int view_w = GetScreenWidth();
        int view_h = GetScreenHeight();
        DrawTexturePro(texture,
                       (Rectangle){0, 0, MAX_RENDER_W, MAX_RENDER_H},
                       (Rectangle){0, 0, (float)view_w, (float)view_h},
                       (Vector2){0, 0}, 0.0f, WHITE);
        draw_hud(render_ms, show_help, gpu != NULL);
        EndDrawing();
        if (!startup_screenshot_saved) {
            TakeScreenshot("world_viewer.png");
            startup_screenshot_saved = 1;
            TraceLog(LOG_INFO, "Saved startup screenshot to world_viewer.png");
        }
        if (IsKeyPressed(KEY_F2)) {
            TakeScreenshot("world_viewer.png");
            TraceLog(LOG_INFO, "Saved world_viewer.png");
        }
    }

    UnloadTexture(texture);
    GpuRenderer_Destroy(gpu);
    CloseWindow();
    free(rgba);
    free(rgb);
    return 0;
}
