#ifndef HYBRID_H
#define HYBRID_H

#include "tracer.h"

/* Cooperative CPU + OpenCL tile rendering (Phase 2). The GPU pops the first
 * N tiles from the shared Morton queue as one rt_tiles batch; OpenMP CPU
 * workers trace the remainder concurrently. The final join waits for the
 * GPU once, then the shared CPU post chain tone maps the HDR film. */

/* Lazily create the OpenCL device (once). Returns 0 if no usable device. */
int rt_hybrid_prepare(int width, int height);

/* Render width x height into the HDR film; CPU post chain runs separately
 * via rt_postprocess_hdr. Returns 0 (film untouched) if the GPU device
 * turned out to be a net loss and was auto-disabled; callers then fall
 * back to the pure CPU path. frame_ms optionally receives the total. */
int rt_render_hybrid_hdr(const Scene *scene, V3 *hdr, int width, int height,
                         int spp, int max_depth, double *frame_ms);

void rt_hybrid_shutdown(void);

#endif
