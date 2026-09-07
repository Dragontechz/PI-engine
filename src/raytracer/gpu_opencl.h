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

/* Returns 1 on success, 0 on failure (caller falls back).
 * Traces at (internal_w x internal_h) and tone maps on GPU; when
 * internal differs from output, the sRGB film is upscaled to
 * (width x height) with Catmull-Rom + CAS sharpening (amount cas). */
int Ocl_Render(OclRenderer *gpu, const Scene *scene, unsigned char *rgb,
               int width, int height, int internal_w, int internal_h,
               int spp, float cas);

void Ocl_Destroy(OclRenderer *gpu);

#endif
