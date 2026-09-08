/* CPU A/B: brute-force traversal vs binned SAH BVH, same scene/camera/SPP.
 * Both paths must produce pixel-identical images (same closest hits, same
 * per-pixel RNG); reports timings and max delta (optimise.txt quality gate).
 * Usage: cpu_ab [small|world] [--spp N] [--width W --height H]
 * Writes ab_naive.bmp and ab_bvh.bmp for visual diffs. */
#include "tracer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

static double now_ms(void) {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static int write_bmp(const char *path, const unsigned char *rgb, int w, int h) {
    int rowBytes = (w * 3 + 3) & ~3;
    int dataSize = rowBytes * h;
    int fileSize = 54 + dataSize;
    unsigned char hdr[54] = { 0 };
    hdr[0] = 'B'; hdr[1] = 'M';
    hdr[2] = (unsigned char)fileSize; hdr[3] = (unsigned char)(fileSize >> 8);
    hdr[4] = (unsigned char)(fileSize >> 16); hdr[5] = (unsigned char)(fileSize >> 24);
    hdr[10] = 54;
    hdr[14] = 40;
    hdr[18] = (unsigned char)w; hdr[19] = (unsigned char)(w >> 8);
    hdr[20] = (unsigned char)(w >> 16); hdr[21] = (unsigned char)(w >> 24);
    hdr[22] = (unsigned char)h; hdr[23] = (unsigned char)(h >> 8);
    hdr[24] = (unsigned char)(h >> 16); hdr[25] = (unsigned char)(h >> 24);
    hdr[26] = 1; hdr[28] = 24;
    FILE *f = fopen(path, "wb");
    if (!f) return 0;
    fwrite(hdr, 1, 54, f);
    unsigned char *row = (unsigned char *)calloc((size_t)rowBytes, 1);
    for (int y = h - 1; y >= 0; y--) {
        for (int x = 0; x < w; x++) {
            int i = (y * w + x) * 3;
            row[x * 3 + 0] = rgb[i + 2];
            row[x * 3 + 1] = rgb[i + 1];
            row[x * 3 + 2] = rgb[i + 0];
        }
        fwrite(row, 1, rowBytes, f);
    }
    free(row);
    fclose(f);
    return 1;
}

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);
    int spp = 4;
    int W = 640, H = 360;
    const char *which = "small";
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--spp") == 0 && i + 1 < argc) spp = atoi(argv[++i]);
        else if (strcmp(argv[i], "--width") == 0 && i + 1 < argc) W = atoi(argv[++i]);
        else if (strcmp(argv[i], "--height") == 0 && i + 1 < argc) H = atoi(argv[++i]);
        else if (strcmp(argv[i], "small") == 0 || strcmp(argv[i], "world") == 0) which = argv[i];
    }
    if (spp < 1) spp = 1;

    static Scene scene; /* Scene embeds large fixed arrays: keep off the stack */
    if (strcmp(which, "world") == 0) scene_world(&scene);
    else scene_small(&scene);

    unsigned char *a = (unsigned char *)malloc((size_t)W * H * 3);
    unsigned char *b = (unsigned char *)malloc((size_t)W * H * 3);
    if (!a || !b) { fprintf(stderr, "OOM\n"); return 1; }

    /* Freeze temporal adaptation so both renders use identical exposure. */
    rt_set_frame_delta(0.0f);

    /* --- BVH run (binned SAH, object + mesh levels) --- */
    double tb0 = now_ms();
    rt_build_accel(&scene);
    double build_ms = now_ms() - tb0;
    int bvh_nodes = scene.accel_node_count;
    printf("SAH build %.2f ms | object nodes %d\n", build_ms, bvh_nodes);

    double t0 = now_ms();
    if (!rt_render_native(&scene, b, W, H, spp, 4)) {
        fprintf(stderr, "BVH render failed\n"); return 1;
    }
    double t1 = now_ms();

    /* --- Naive run: brute force = no object BVH, no mesh BVH --- */
    int mesh_nodes[RT_MAX_MESHES];
    for (int i = 0; i < scene.mesh_count; i++) {
        mesh_nodes[i] = scene.meshes[i].node_count;
        scene.meshes[i].node_count = 0;
    }
    scene.accel_node_count = 0;
    for (int i = 0; i < scene.mesh_count; i++)
        printf("mesh %d: %d SAH nodes\n", i, mesh_nodes[i]);
    double t2 = now_ms();
    if (!rt_render_native(&scene, a, W, H, spp, 4)) {
        fprintf(stderr, "naive render failed\n"); return 1;
    }
    double t3 = now_ms();

    double mae = 0.0;
    int maxd = 0;
    long npx = (long)W * H;
    for (long i = 0; i < npx * 3; i++) {
        int d = abs((int)a[i] - (int)b[i]);
        mae += d;
        if (d > maxd) maxd = d;
    }
    mae /= (double)(npx * 3);
    printf("scene=%s spp=%d %dx%d\n", which, spp, W, H);
    printf("naive %.0f ms | SAH BVH %.0f ms | speedup %.2fx\n",
           t3 - t2, t1 - t0, (t3 - t2) / (t1 - t0));
    printf("image delta: MAE %.4f/255 | max %d %s\n",
           mae, maxd, maxd == 0 ? "(pixel-identical)" : "(MISMATCH)");
    if (!write_bmp("ab_naive.bmp", a, W, H) || !write_bmp("ab_bvh.bmp", b, W, H))
        fprintf(stderr, "BMP write failed\n");
    else
        printf("wrote ab_naive.bmp + ab_bvh.bmp\n");
    free(a); free(b);
    return maxd == 0 ? 0 : 3;
}
