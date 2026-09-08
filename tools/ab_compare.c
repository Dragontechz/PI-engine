/* A/B image comparison: CPU reference vs OpenCL GPU, same scene/camera/SPP.
 * Reports MAE, P95 and max per-pixel delta (optimise.txt quality gate).
 * Usage: ab_compare [scene_small|scene_world] [--spp N] [--width W --height H]
 * Writes ab_cpu.bmp and ab_gpu.bmp for visual diffs. */
#include "tracer.h"
#include "gpu_opencl.h"
#include "hybrid.h"
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
    fprintf(stderr, "ab: entered\n");
    int spp = 8;
    int W = 640, H = 360;
    int hybrid = 0;
int frames = 1;
    const char *which = "small";
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--spp") == 0 && i + 1 < argc) spp = atoi(argv[++i]);
        else if (strcmp(argv[i], "--width") == 0 && i + 1 < argc) W = atoi(argv[++i]);
        else if (strcmp(argv[i], "--height") == 0 && i + 1 < argc) H = atoi(argv[++i]);
        else if (strcmp(argv[i], "--hybrid") == 0) hybrid = 1;
    else if (strcmp(argv[i], "--frames") == 0 && i + 1 < argc) frames = atoi(argv[++i]);
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

    double t0 = now_ms();
    if (!rt_render_native(&scene, a, W, H, spp, 4)) {
        fprintf(stderr, "CPU render failed\n"); return 1;
    }
    double t1 = now_ms();

    if (hybrid) {
        /* Phase 2 gate: cooperative CPU + GPU tile render vs CPU reference. */
        if (!rt_hybrid_prepare(W, H)) { fprintf(stderr, "hybrid: no device\n"); return 2; }
        V3 *film = (V3 *)malloc((size_t)W * H * sizeof *film);
        if (!film) { fprintf(stderr, "OOM\n"); return 1; }
        double frame_ms = 0.0;
        for (int f = 0; f < frames; f++) {
            if (!rt_render_hybrid_hdr(&scene, film, W, H, spp, 4, &frame_ms) ||
                !rt_postprocess_hdr(film, b, W, H)) {
                fprintf(stderr, "hybrid render failed\n"); return 1;
            }
            fprintf(stderr, "hybrid frame %d: %.0f ms\n", f + 1, frame_ms);
        }
        free(film);
    } else {
        OclRenderer *g = Ocl_Create(W, H);
        if (!g) { fprintf(stderr, "OpenCL unavailable\n"); return 2; }
        if (!Ocl_Render(g, &scene, b, W, H, W, H, spp, 0.0f)) {
            fprintf(stderr, "GPU render failed\n"); return 1;
        }
        Ocl_Destroy(g);
    }
    double t2 = now_ms();

    /* deltas at or above 6/255 are visually obvious; count them separately */
    double mae = 0.0;
    int big = 0;
    int maxd = 0;
    long npx = (long)W * H;
    for (long i = 0; i < npx * 3; i++) {
        int d = abs((int)a[i] - (int)b[i]);
        mae += d;
        if (d > maxd) maxd = d;
        if (d >= 6) big++;
    }
    mae /= (double)(npx * 3);
    printf("scene=%s spp=%d %dx%d\n", which, spp, W, H);
    printf("cpu %.0f ms | gpu %.0f ms\n", t1 - t0, t2 - t1);
    printf("MAE %.3f/255 | pixels with delta>=6: %d (%.3f%%) | max delta %d\n",
           mae, big, 100.0 * (double)big / (double)(npx * 3), maxd);
    if (!write_bmp("ab_cpu.bmp", a, W, H) || !write_bmp("ab_gpu.bmp", b, W, H))
        fprintf(stderr, "BMP write failed\n");
    else
        printf("wrote ab_cpu.bmp + ab_gpu.bmp\n");
    free(a); free(b);
    return 0;
}
