#ifndef GPU_RENDERER_H
#define GPU_RENDERER_H

#include "tracer.h"

/* GLSL (raylib rlgl, main thread) renderer for the fast packet look:
 * primary rays + point-light shadows, bunny via triangle texture.
 * Falls back to the CPU paths when unavailable. */
typedef struct GpuRenderer GpuRenderer;

/* Must be called after InitWindow. Returns NULL when GL/shader init fails. */
GpuRenderer *GpuRenderer_Create(int width, int height);

/* Returns 1 on success, 0 when the GPU path failed (caller should fall back
 * to CPU; the renderer marks itself broken after a failure).
 * internal_w/internal_h <= 0 means "same as width/height" (no upscale). */
int GpuRenderer_Render(GpuRenderer *gpu, const Scene *scene, unsigned char *rgb,
                       int width, int height, int internal_w, int internal_h,
                       int spp, float cas);

int GpuRenderer_RenderAccum(GpuRenderer *gpu, const Scene *scene, unsigned char *rgb,
                            int width, int height, int internal_w, int internal_h,
                            int spp, float cas, int reset);

void GpuRenderer_Destroy(GpuRenderer *gpu);

/* Share raylib's texture with the OpenCL device for zero-copy presentation
 * (no readback, no UpdateTexture). Returns 0 when interop is unavailable. */
int GpuRenderer_AttachGlTexture(GpuRenderer *gpu, unsigned tex_id, int width, int height);

/* Non-zero while Ocl_Render presents straight into the attached texture. */
int GpuRenderer_GlInterop(GpuRenderer *gpu);

#endif
