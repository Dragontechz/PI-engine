/* hybrid.c — cooperative CPU + OpenCL tile scheduler.
 *
 * Both engines pull from one Morton-ordered tile list: the GPU renders one
 * rt_tiles batch (4..16 tiles, one workgroup per 16x16 tile) asynchronously
 * while OpenMP CPU workers trace the rest. The only CPU stall is the final
 * join (Ocl_ReadTiles). A feedback controller grows/shrinks the GPU batch
 * toward load balance, and the GPU is auto-disabled if it is a net loss. */
#include "hybrid.h"

#include "gpu_opencl.h"

#include <stdio.h>
#include <stdlib.h>
#include <windows.h>

#define RT_HYBRID_TILE 16
#define RT_HYBRID_BATCH_MIN 4
#define RT_HYBRID_BATCH_MAX 16

struct HybridState {
    OclRenderer *gpu;
    int device_ok;        /* device created, not broken, not benched off */
    int batch;            /* GPU batch size in tiles */
    int measured;         /* hybrid frames with both engines busy */
    int slow_streak;      /* consecutive frames slower than CPU-only */
    int frames;           /* hybrid frames since start */
    double cpu_rate;      /* EMA of CPU tile throughput (tiles/ms) */
    double hybrid_ms;     /* EMA of total hybrid frame time */
    double cpu_only_ms;   /* EMA of projected CPU-only frame time */
};

static struct HybridState g_hyb;

static double ema_step(double old, double v, double alpha) {
    return old < 0.0 ? v : old + alpha * (v - old);
}

static double qpc_ms(void) {
    LARGE_INTEGER f, t;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&t);
    return (double)t.QuadPart * 1000.0 / (double)f.QuadPart;
}

int rt_hybrid_prepare(int width, int height) {
    if (g_hyb.gpu) return g_hyb.device_ok;
    if (width > 1280 || height > 720) return 0; /* OCL_MAX_W/H */
    g_hyb.gpu = Ocl_Create(width, height);
    g_hyb.device_ok = g_hyb.gpu != NULL;
    g_hyb.batch = 8; /* spec: GPU pops 4..16 tiles per batch */
    fprintf(stderr, "Hybrid: OpenCL device %s\n",
            g_hyb.device_ok ? "ready" : "unavailable, CPU-only");
    return g_hyb.device_ok;
}

void rt_hybrid_shutdown(void) {
    if (g_hyb.gpu) {
        Ocl_Destroy(g_hyb.gpu);
        g_hyb.gpu = NULL;
    }
    g_hyb.device_ok = 0;
}

int rt_render_hybrid_hdr(const Scene *s, V3 *hdr, int width, int height,
                         int spp, int max_depth, double *frame_ms) {
    if (!s || !hdr || width <= 0 || height <= 0) return 0;
    const int tile_count = rt_build_tile_origins(width, height, RT_HYBRID_TILE, NULL, 0);
    if (tile_count <= 0) return 0;

    /* Pure CPU when no device: identical output to rt_render_native. */
    if (!g_hyb.device_ok || !g_hyb.gpu ||
        tile_count < RT_HYBRID_BATCH_MIN + RT_HYBRID_BATCH_MIN) {
        int *tiles = (int *)malloc((size_t)tile_count * 2 * sizeof *tiles);
        if (!tiles) return 0;
        rt_build_tile_origins(width, height, RT_HYBRID_TILE, tiles, tile_count);
        double t0 = qpc_ms();
        int ok = rt_render_tiles_hdr(s, hdr, width, height, spp, max_depth,
                                     tiles, tile_count);
        double ms = qpc_ms() - t0;
        g_hyb.cpu_rate = ema_step(g_hyb.cpu_rate, tile_count / ms, 0.15);
        g_hyb.cpu_only_ms = ema_step(g_hyb.cpu_only_ms, ms, 0.15);
        if (frame_ms) *frame_ms = ms;
        free(tiles);
        return ok;
    }

    OclRenderer *gpu = g_hyb.gpu;
    int batch = g_hyb.batch;
    if (batch < RT_HYBRID_BATCH_MIN) batch = RT_HYBRID_BATCH_MIN;
    if (batch > RT_HYBRID_BATCH_MAX) batch = RT_HYBRID_BATCH_MAX;
    if (batch > tile_count - RT_HYBRID_BATCH_MIN) batch = tile_count - RT_HYBRID_BATCH_MIN;

    int *tiles = (int *)malloc((size_t)tile_count * 2 * sizeof *tiles);
    if (!tiles) return 0;
    rt_build_tile_origins(width, height, RT_HYBRID_TILE, tiles, tile_count);

    /* GPU takes the head of the Morton queue, CPU the remainder. */
    const double t0 = qpc_ms();
    if (!Ocl_TraceTiles(gpu, s, width, height, spp, tiles, batch, RT_HYBRID_TILE)) {
        free(tiles);
        fprintf(stderr, "Hybrid: device failed, falling back to CPU\n");
        g_hyb.device_ok = 0;
        return 0;
    }
    const int cpu_tiles = tile_count - batch;
    double cpu_ms = 0.0;
    if (cpu_tiles > 0) {
        double tc0 = qpc_ms();
        rt_render_tiles_hdr(s, hdr, width, height, spp, max_depth,
                            tiles + batch * 2, cpu_tiles);
        cpu_ms = qpc_ms() - tc0;
    }
    double gpu_tail = 0.0;
    int ok = Ocl_ReadTiles(gpu, hdr, width, height, tiles, batch, &gpu_tail);
    const double total = qpc_ms() - t0;
    free(tiles);
    if (!ok) {
        fprintf(stderr, "Hybrid: device failed at join, falling back to CPU\n");
        g_hyb.device_ok = 0;
        return 0;
    }

    /* Controller: balance the GPU tail against the CPU phase. A near-zero
     * tail means the GPU starved (grow the batch); a tail past the CPU
     * phase means it is overfed (shrink). */
    g_hyb.cpu_rate = ema_step(g_hyb.cpu_rate, cpu_tiles / (cpu_ms > 0.01 ? cpu_ms : 0.01), 0.15);
    g_hyb.batch = batch;
    if (gpu_tail < cpu_ms * 0.25) g_hyb.batch += 2;
    else if (gpu_tail > cpu_ms) g_hyb.batch -= 2;
    if (g_hyb.batch < RT_HYBRID_BATCH_MIN) g_hyb.batch = RT_HYBRID_BATCH_MIN;
    if (g_hyb.batch > RT_HYBRID_BATCH_MAX) g_hyb.batch = RT_HYBRID_BATCH_MAX;

    /* Auto-disable: if the hybrid frame is consistently slower than the
     * CPU-only projection (device too weak, e.g. iGPU vs 4 threads), stop
     * using it and let the pure CPU path take over. */
    g_hyb.measured++;
    g_hyb.hybrid_ms = ema_step(g_hyb.hybrid_ms, total, 0.15);
    g_hyb.cpu_only_ms = ema_step(g_hyb.cpu_only_ms,
                                 tile_count / (g_hyb.cpu_rate > 0.01 ? g_hyb.cpu_rate : 0.01), 0.15);
    if (g_hyb.hybrid_ms > g_hyb.cpu_only_ms * 1.05) g_hyb.slow_streak++;
    else g_hyb.slow_streak = 0;
    if (g_hyb.measured >= 5 && g_hyb.slow_streak >= 3) {
        fprintf(stderr, "Hybrid: GPU is a net loss (%.1f ms vs %.1f ms CPU-only), disabling\n",
                g_hyb.hybrid_ms, g_hyb.cpu_only_ms);
        g_hyb.device_ok = 0;
    }

    if (++g_hyb.frames % 30 == 0 || g_hyb.frames <= 3) {
        fprintf(stderr, "Hybrid: frame %d batch %d gpu tail %.1f cpu %.1f total %.1f ms (cpu-only ~%.1f)\n",
                g_hyb.frames, batch, gpu_tail, cpu_ms, total, g_hyb.cpu_only_ms);
        fflush(stderr);
    }
    if (frame_ms) *frame_ms = total;
    return 1;
}
