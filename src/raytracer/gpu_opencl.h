#ifndef GPU_OPENCL_H
#define GPU_OPENCL_H

#include "tracer.h"

/* OpenCL compute renderer: runs the packet path tracer (primary rays +
 * point-light shadows + bunny triangles) on the GPU through OpenCL,
 * loaded dynamically from OpenCL.dll at runtime (no SDK required).
 * Falls back gracefully: create/render return NULL/0 when OpenCL is
 * unavailable or fails. */
typedef struct OclRenderer OclRenderer;

/* May be called before or after InitWindow (no GL interop used). */
OclRenderer *Ocl_Create(int width, int height);

/* Share raylib's GL texture with the OpenCL device (cl_khr_gl_sharing). On
 * success Ocl_Render presents the final frame into this texture directly
 * and skips the CPU readback entirely. Returns 0 when interop is
 * unavailable (caller keeps using the rgb readback + UpdateTexture). */
int Ocl_AttachGlTexture(OclRenderer *gpu, unsigned tex_id, int width, int height);

/* Non-zero while a GL texture is attached and used for presentation. */
int Ocl_GlInterop(const OclRenderer *gpu);

/* Returns 1 on success, 0 on failure (caller falls back).
 * Traces at (internal_w x internal_h) and tone maps on GPU; when
 * internal differs from output, the sRGB film is upscaled to
 * (width x height) with Catmull-Rom + CAS sharpening (amount cas). */
int Ocl_Render(OclRenderer *gpu, const Scene *scene, unsigned char *rgb,
               int width, int height, int internal_w, int internal_h,
               int spp, float cas);

/* Hybrid tile workers: enqueue a kernel covering the listed tile origins
 * ((tx, ty) pairs, tile x tile work-items each; tile is 8 or 16) without
 * blocking, then later finish + read back only those tiles into hdr
 * (width x height V3 pixels). *kernel_ms receives GPU time in ms. */
int Ocl_TraceTiles(OclRenderer *gpu, const Scene *scene, int width, int height,
                   int spp, const int *tiles_xy, int tile_count, int tile);
int Ocl_ReadTiles(OclRenderer *gpu, V3 *hdr, int width, int height,
                  const int *tiles_xy, int tile_count, double *kernel_ms);

void Ocl_Destroy(OclRenderer *gpu);

#endif
