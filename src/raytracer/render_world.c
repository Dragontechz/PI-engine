/* Driver: renders a scene to render.bmp + render.png.
 * Usage: tracer_render [block|world] [samples] */
#include "tracer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
    const char *which = "world";
    const char *mode = "adaptive";
    int adaptive_res = 0;
    int samples = 1, spp_min = 4, spp_max = 16;
    int W = 1280, H = 720;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "block") == 0 || strcmp(argv[i], "world") == 0) which = argv[i];
        else if (strcmp(argv[i], "--mode") == 0 && i + 1 < argc) mode = argv[++i];
        else if (strcmp(argv[i], "--adaptive-res") == 0) adaptive_res = 1;
        else if (strcmp(argv[i], "--spp") == 0 && i + 1 < argc) samples = atoi(argv[++i]);
        else if (strcmp(argv[i], "--spp-min") == 0 && i + 1 < argc) spp_min = atoi(argv[++i]);
        else if (strcmp(argv[i], "--spp-max") == 0 && i + 1 < argc) spp_max = atoi(argv[++i]);
        else if (strcmp(argv[i], "--width") == 0 && i + 1 < argc) W = atoi(argv[++i]);
        else if (strcmp(argv[i], "--height") == 0 && i + 1 < argc) H = atoi(argv[++i]);
        else if (i == 1) samples = atoi(argv[i]);
        else if (i == 2) W = atoi(argv[i]);
        else if (i == 3) H = atoi(argv[i]);
    }

    /* Scene contains large fixed-size mesh/BVH arrays and must not live on the
     * small Windows thread stack. */
    static Scene scene;
    if (strcmp(which, "block") == 0) scene_block(&scene);
    else scene_world(&scene);

    printf("scene '%s': mode=%s adaptive_res=%d samples=%d adaptive=%d..%d, %dx%d\n",
           which, mode, adaptive_res, samples, spp_min, spp_max, W, H);

    unsigned char *img = (unsigned char *)malloc((size_t)W * H * 3);
    if (!img) { printf("OOM\n"); return 1; }
    int rendered;
    if (strcmp(mode, "native") == 0) {
        rendered = rt_render_native(&scene, img, W, H, samples, 4);
    } else if (adaptive_res || strcmp(mode, "upscale") == 0) {
        rendered = rt_render_adaptive_fast(&scene, img, W, H, spp_min, spp_max);
    } else {
        rendered = rt_render_adaptive(&scene, img, W, H, spp_min, spp_max, 0.05f, 0.20f, 4);
    }
    if (!rendered) {
        printf("RENDER FAIL\n");
        free(img);
        return 1;
    }
    if (!write_bmp("render.bmp", img, W, H)) {
        printf("BMP FAIL\n");
        free(img);
        return 1;
    }
    printf("RENDER OK: render.bmp\n");
    free(img);
    return 0;
}
