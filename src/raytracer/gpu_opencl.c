/* OpenCL renderer, loaded dynamically from OpenCL.dll (ships with the
 * Intel graphics driver). No OpenCL SDK needed: the handful of entry
 * points we use are declared here and resolved with GetProcAddress. */
#include "gpu_opencl.h"
#include "gpu_opencl_kernel.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#define OCL_MAX_W 1280
#define OCL_MAX_H 720
#define OCL_MAX_SPHERES 24
#define OCL_MAX_BOXES 24
#define OCL_MAX_CYLS 24
#define OCL_MAX_PLANES 8
#define OCL_MAX_LIGHTS RT_MAX_LIGHTS
#define OCL_MAX_TRIS 4096
#define OCL_GRID_CELLS 1000  /* 10 x 10 x 10 */

/* Per-frame scalar block layout — must mirror F_* in gpu_opencl_kernel.cl. */
#define F_CAM 0
#define F_FWD 4
#define F_RIGHT 8
#define F_UP 12
#define F_ASPECT 16
#define F_W 20
#define F_H 21
#define F_SPP 22
#define F_SAMPLE_BASE 23
#define F_COUNTS 24
#define F_TRI 28
#define F_LIGHTS 29
#define F_FOG 32
#define F_SUN_DIR 36
#define F_SUN_COL 40
#define F_BMAT 44
#define F_BEMI 48
#define F_ACCUM 30
#define FRAME_FLOATS 52
#define OCL_GRID_TRI_CAP (OCL_MAX_TRIS * 8)  /* max CSR triangle refs */
#define OCL_LOG_GROUPS 256   /* rt_logavg partial sums */
#define OCL_LOG_LOCAL 64     /* rt_logavg local size */
#define OCL_MAX_TILES 16384  /* tile batch cap (fits 8x8 tiles at 1280x720) */

/* ---------------------------------------------------------------- API decls */
#ifdef _WIN32
#define OCL_CALL __stdcall
#else
#define OCL_CALL
#endif

typedef int cl_int;
typedef unsigned int cl_uint;
typedef unsigned long long cl_ulong;
typedef void *clh; /* platform/device/context/queue/program/kernel/mem */

#define CL_SUCCESS 0
#define CL_TRUE 1
#define CL_FALSE 0
#define CL_MEM_READ_ONLY (1 << 2)
#define CL_MEM_WRITE_ONLY (1 << 1)
#define CL_MEM_READ_WRITE (1 << 0)
#define CL_MEM_COPY_HOST_PTR (1 << 5)
#define CL_DEVICE_TYPE_GPU (1 << 2)
#define CL_PLATFORM_NAME 0x0902
#define CL_DEVICE_NAME 0x102B
#define CL_PROGRAM_BUILD_LOG 0x1183
#define CL_GL_CONTEXT_KHR 0x2008
#define CL_WGL_HDC_KHR 0x200B
#define CL_GL_TEXTURE_2D 0x0DE1 /* GL_TEXTURE_2D */

typedef cl_int (OCL_CALL *pfn_clGetPlatformIDs)(cl_uint, clh *, cl_uint *);
typedef cl_int (OCL_CALL *pfn_clGetDeviceIDs)(clh, cl_ulong, cl_uint, clh *, cl_uint *);
typedef clh (OCL_CALL *pfn_clCreateContext)(const intptr_t *, cl_uint, const clh *,
                                            void (*)(const char *, const void *, size_t, void *),
                                            void *, cl_int *);
typedef clh (OCL_CALL *pfn_clCreateCommandQueue)(clh, clh, cl_ulong, cl_int *);
typedef clh (OCL_CALL *pfn_clCreateBuffer)(clh, cl_ulong, size_t, void *, cl_int *);
typedef clh (OCL_CALL *pfn_clCreateProgramWithSource)(clh, cl_uint, const char **,
                                                      const size_t *, cl_int *);
typedef cl_int (OCL_CALL *pfn_clBuildProgram)(clh, cl_uint, const clh *, const char *,
                                              void (*)(clh, void *), void *);
typedef cl_int (OCL_CALL *pfn_clGetProgramBuildInfo)(clh, clh, cl_uint, size_t, void *, size_t *);
typedef clh (OCL_CALL *pfn_clCreateKernel)(clh, const char *, cl_int *);
typedef cl_int (OCL_CALL *pfn_clSetKernelArg)(clh, cl_uint, size_t, const void *);
typedef cl_int (OCL_CALL *pfn_clEnqueueNDRangeKernel)(clh, clh, cl_uint, const size_t *,
                                                      const size_t *, const size_t *,
                                                      cl_uint, const void *, void *);
typedef cl_int (OCL_CALL *pfn_clEnqueueWriteBuffer)(clh, clh, cl_uint, size_t, size_t,
                                                    const void *, cl_uint, const void *, void *);
typedef cl_int (OCL_CALL *pfn_clEnqueueReadBuffer)(clh, clh, cl_uint, size_t, size_t, void *,
                                                   cl_uint, const void *, void *);
typedef cl_int (OCL_CALL *pfn_clFinish)(clh);
typedef cl_int (OCL_CALL *pfn_clGetPlatformInfo)(clh, cl_uint, size_t, void *, size_t *);
typedef cl_int (OCL_CALL *pfn_clGetDeviceInfo)(clh, cl_uint, size_t, void *, size_t *);
typedef cl_int (OCL_CALL *pfn_clGetKernelWorkGroupInfo)(clh, clh, cl_uint, size_t, void *, size_t *);
typedef cl_int (OCL_CALL *pfn_clReleaseMemObject)(clh);
typedef cl_int (OCL_CALL *pfn_clReleaseKernel)(clh);
typedef cl_int (OCL_CALL *pfn_clReleaseProgram)(clh);
typedef cl_int (OCL_CALL *pfn_clReleaseCommandQueue)(clh);
typedef cl_int (OCL_CALL *pfn_clReleaseContext)(clh);
typedef clh (OCL_CALL *pfn_clCreateFromGLTexture)(clh, cl_ulong, cl_uint, int, cl_uint, cl_int *);
typedef cl_int (OCL_CALL *pfn_clEnqueueAcquireGLObjects)(clh, cl_uint, const clh *,
                                                         cl_uint, const void *, void *);
typedef cl_int (OCL_CALL *pfn_clEnqueueReleaseGLObjects)(clh, cl_uint, const clh *,
                                                         cl_uint, const void *, void *);

struct OclApi {
    pfn_clGetPlatformIDs GetPlatformIDs;
    pfn_clGetDeviceIDs GetDeviceIDs;
    pfn_clCreateContext CreateContext;
    pfn_clCreateCommandQueue CreateCommandQueue;
    pfn_clCreateBuffer CreateBuffer;
    pfn_clCreateProgramWithSource CreateProgramWithSource;
    pfn_clBuildProgram BuildProgram;
    pfn_clGetProgramBuildInfo GetProgramBuildInfo;
    pfn_clCreateKernel CreateKernel;
    pfn_clSetKernelArg SetKernelArg;
    pfn_clEnqueueNDRangeKernel EnqueueNDRangeKernel;
    pfn_clEnqueueWriteBuffer EnqueueWriteBuffer;
    pfn_clEnqueueReadBuffer EnqueueReadBuffer;
    pfn_clFinish Finish;
    pfn_clGetPlatformInfo GetPlatformInfo;
    pfn_clGetDeviceInfo GetDeviceInfo;
    pfn_clGetKernelWorkGroupInfo GetKernelWorkGroupInfo;
    pfn_clReleaseMemObject ReleaseMemObject;
    pfn_clReleaseKernel ReleaseKernel;
    pfn_clReleaseProgram ReleaseProgram;
    pfn_clReleaseCommandQueue ReleaseCommandQueue;
    pfn_clReleaseContext ReleaseContext;
    pfn_clCreateFromGLTexture CreateFromGLTexture;
    pfn_clEnqueueAcquireGLObjects EnqueueAcquireGLObjects;
    pfn_clEnqueueReleaseGLObjects EnqueueReleaseGLObjects;
};

struct OclRenderer {
    struct OclApi api;
    clh device;                 /* cl_device_id for kernel workgroup queries */
    clh ctx, queue, prog, kernel;
    clh k_logavg, k_post, k_upscale, k_cas, k_tiles, k_present;
    clh d_out;
    clh d_gl;                   /* GL-shared presentation image (interop) */
    clh d_tilebuf;
    clh d_sph, d_sph_mat, d_sph_emi, d_sph_texA, d_sph_texB;
    clh d_box_min, d_box_max, d_box_mat, d_box_emi, d_box_texA, d_box_texB;
    clh d_cyl_b, d_cyl_h, d_cyl_mat, d_cyl_emi, d_cyl_texA, d_cyl_texB;
    clh d_plane_pos, d_plane_mat, d_plane_emi, d_plane_texA, d_plane_texB;
    clh d_lpos, d_lcol, d_lrad, d_tris;
    clh d_grid_a, d_grid_dims, d_grid_off, d_grid_tri;
    const Scene *uploaded_scene; /* static geometry uploaded for this scene */
    clh d_rgb_in, d_present, d_cas, d_logpart;
    clh d_frame;
    int tile_size;              /* rt_tiles workgroup tile (8 or 16), 0 = unset */
    float h_grid_a[4];          /* origin.xyz, cell size */
    unsigned h_grid_dims[4];    /* nx, ny, nz, 0 */
    unsigned h_grid_off[OCL_GRID_CELLS + 1];
    unsigned *h_grid_tri;       /* CSR triangle indices */
    int h_grid_tri_len;
    int tris_uploaded;
    int tri_count;
    int frames;
    int accum_samples;
    int broken;
};

/* --------------------------------------------------------------------- init */
static int resolve_api(struct OclApi *api, HMODULE mod) {
    struct { const char *name; void **target; } table[] = {
        { "clGetPlatformIDs", (void **)&api->GetPlatformIDs },
        { "clGetDeviceIDs", (void **)&api->GetDeviceIDs },
        { "clCreateContext", (void **)&api->CreateContext },
        { "clCreateCommandQueue", (void **)&api->CreateCommandQueue },
        { "clCreateBuffer", (void **)&api->CreateBuffer },
        { "clCreateProgramWithSource", (void **)&api->CreateProgramWithSource },
        { "clBuildProgram", (void **)&api->BuildProgram },
        { "clGetProgramBuildInfo", (void **)&api->GetProgramBuildInfo },
        { "clCreateKernel", (void **)&api->CreateKernel },
        { "clSetKernelArg", (void **)&api->SetKernelArg },
        { "clEnqueueNDRangeKernel", (void **)&api->EnqueueNDRangeKernel },
        { "clEnqueueWriteBuffer", (void **)&api->EnqueueWriteBuffer },
        { "clEnqueueReadBuffer", (void **)&api->EnqueueReadBuffer },
        { "clFinish", (void **)&api->Finish },
        { "clGetPlatformInfo", (void **)&api->GetPlatformInfo },
        { "clGetDeviceInfo", (void **)&api->GetDeviceInfo },
        { "clGetKernelWorkGroupInfo", (void **)&api->GetKernelWorkGroupInfo },
        { "clReleaseMemObject", (void **)&api->ReleaseMemObject },
        { "clReleaseKernel", (void **)&api->ReleaseKernel },
        { "clReleaseProgram", (void **)&api->ReleaseProgram },
        { "clReleaseCommandQueue", (void **)&api->ReleaseCommandQueue },
        { "clReleaseContext", (void **)&api->ReleaseContext },
    };
    for (size_t i = 0; i < sizeof table / sizeof table[0]; i++) {
        *table[i].target = (void *)GetProcAddress(mod, table[i].name);
        if (!*table[i].target) return 0;
    }
    api->CreateFromGLTexture = (pfn_clCreateFromGLTexture)GetProcAddress(mod, "clCreateFromGLTexture");
    api->EnqueueAcquireGLObjects = (pfn_clEnqueueAcquireGLObjects)GetProcAddress(mod, "clEnqueueAcquireGLObjects");
    api->EnqueueReleaseGLObjects = (pfn_clEnqueueReleaseGLObjects)GetProcAddress(mod, "clEnqueueReleaseGLObjects");
    return 1;
}

static void release_all(struct OclRenderer *g) {
    if (!g) return;
    if (g->api.ReleaseMemObject) {
        clh *mems[] = { &g->d_out, &g->d_sph, &g->d_sph_mat, &g->d_sph_emi,
                        &g->d_sph_texA, &g->d_sph_texB,
                        &g->d_box_min, &g->d_box_max, &g->d_box_mat, &g->d_box_emi,
                        &g->d_box_texA, &g->d_box_texB,
                        &g->d_cyl_b, &g->d_cyl_h, &g->d_cyl_mat, &g->d_cyl_emi,
                        &g->d_cyl_texA, &g->d_cyl_texB,
                        &g->d_plane_pos, &g->d_plane_mat, &g->d_plane_emi,
                        &g->d_plane_texA, &g->d_plane_texB,
                        &g->d_lpos, &g->d_lcol, &g->d_lrad, &g->d_tris,
                        &g->d_grid_a, &g->d_grid_dims, &g->d_grid_off, &g->d_grid_tri,
                        &g->d_rgb_in, &g->d_present, &g->d_cas, &g->d_logpart,
                         &g->d_tilebuf, &g->d_gl, &g->d_frame };
        for (size_t i = 0; i < sizeof mems / sizeof mems[0]; i++) {
            if (*mems[i]) g->api.ReleaseMemObject(*mems[i]);
        }
    }
    clh *kerns[] = { &g->kernel, &g->k_logavg, &g->k_post, &g->k_upscale, &g->k_cas,
                     &g->k_tiles, &g->k_present };
    for (size_t i = 0; i < sizeof kerns / sizeof kerns[0]; i++) {
        if (*kerns[i] && g->api.ReleaseKernel) g->api.ReleaseKernel(*kerns[i]);
    }
    if (g->prog && g->api.ReleaseProgram) g->api.ReleaseProgram(g->prog);
    if (g->queue && g->api.ReleaseCommandQueue) g->api.ReleaseCommandQueue(g->queue);
    if (g->ctx && g->api.ReleaseContext) g->api.ReleaseContext(g->ctx);
}

static clh make_buffer(struct OclRenderer *g, cl_ulong flags, size_t size) {
    cl_int err = 0;
    clh m = g->api.CreateBuffer(g->ctx, flags, size, NULL, &err);
    if (err != CL_SUCCESS || !m) return NULL;
    return m;
}

static cl_int api_wb(struct OclRenderer *g, clh buf, const void *data, size_t size) {
    return g->api.EnqueueWriteBuffer(g->queue, buf, CL_TRUE, 0, size, data, 0, NULL, NULL);
}

#define OCL_LOG(msg) do { fprintf(stderr, "OpenCL: %s\n", msg); fflush(stderr); } while (0)

OclRenderer *Ocl_Create(int width, int height) {
    (void)width; (void)height; /* buffers are sized for the maximum frame */
    HMODULE mod = LoadLibraryA("OpenCL.dll");
    OCL_LOG("Ocl_Create entered");
    if (!mod) { OCL_LOG("OpenCL.dll load FAILED"); return NULL; }
    OCL_LOG("dll loaded");
    OclRenderer *g = (OclRenderer *)calloc(1, sizeof *g);
    if (!g) return NULL;
    if (!resolve_api(&g->api, mod)) {
        fprintf(stderr, "OpenCL: missing entry points\n");
        free(g);
        return NULL;
    }
    OCL_LOG("api resolved");

    cl_uint nplats = 0;
    if (g->api.GetPlatformIDs(0, NULL, &nplats) != CL_SUCCESS || nplats == 0) {
        fprintf(stderr, "OpenCL: no platforms\n");
        free(g);
        return NULL;
    }
    clh *plats = (clh *)malloc(nplats * sizeof *plats);
    if (!plats || g->api.GetPlatformIDs(nplats, plats, NULL) != CL_SUCCESS) {
        free(plats); free(g); return NULL;
    }

    clh dev = NULL;
    for (cl_uint p = 0; p < nplats && !dev; p++) {
        char pname[256] = { 0 };
        g->api.GetPlatformInfo(plats[p], CL_PLATFORM_NAME, sizeof pname, pname, NULL);
        cl_uint ndev = 0;
        if (g->api.GetDeviceIDs(plats[p], CL_DEVICE_TYPE_GPU, 0, NULL, &ndev) != CL_SUCCESS
            || ndev == 0) continue;
        clh *devs = (clh *)malloc(ndev * sizeof *devs);
        if (!devs) continue;
        if (g->api.GetDeviceIDs(plats[p], CL_DEVICE_TYPE_GPU, ndev, devs, NULL) != CL_SUCCESS) {
            free(devs); continue;
        }
        for (cl_uint d = 0; d < ndev && !dev; d++) {
            char dname[256] = { 0 };
            g->api.GetDeviceInfo(devs[d], CL_DEVICE_NAME, sizeof dname, dname, NULL);
            dev = devs[d];
            fprintf(stderr, "OpenCL: platform '%s' device '%s'\n", pname, dname);
        }
        free(devs);
    }
    free(plats);
    if (!dev) {
        fprintf(stderr, "OpenCL: no GPU device found\n");
        free(g); return NULL;
    }

    cl_int err = 0;
    g->device = dev;
    /* Share with raylib's GL context when one is current on this thread: the
     * final frame can then be presented with zero copies (cl_khr_gl_sharing). */
    intptr_t ctx_props[5] = { 0, 0, 0, 0, 0 };
    typedef void *WINAPI pfn_wgl_ctx(void);
    typedef void *WINAPI pfn_wgl_dc(void);
    static pfn_wgl_ctx *wgl_ctx_fn = NULL;
    static pfn_wgl_dc *wgl_dc_fn = NULL;
    if (!wgl_ctx_fn) {
        HMODULE ogl = GetModuleHandleA("opengl32.dll");
        if (ogl) {
            wgl_ctx_fn = (pfn_wgl_ctx *)GetProcAddress(ogl, "wglGetCurrentContext");
            wgl_dc_fn = (pfn_wgl_dc *)GetProcAddress(ogl, "wglGetCurrentDC");
        }
    }
    HGLRC gl_ctx = wgl_ctx_fn ? (HGLRC)wgl_ctx_fn() : NULL;
    HDC gl_dc = wgl_dc_fn ? (HDC)wgl_dc_fn() : NULL;
    if (gl_ctx && gl_dc) {
        ctx_props[0] = CL_GL_CONTEXT_KHR;
        ctx_props[1] = (intptr_t)gl_ctx;
        ctx_props[2] = CL_WGL_HDC_KHR;
        ctx_props[3] = (intptr_t)gl_dc;
    }
    g->ctx = g->api.CreateContext(ctx_props, 1, &dev, NULL, NULL, &err);
    if ((!g->ctx || err != CL_SUCCESS) && gl_ctx && gl_dc) {
        intptr_t no_gl_props[1] = { 0 };
        err = 0;
        g->ctx = g->api.CreateContext(no_gl_props, 1, &dev, NULL, NULL, &err);
    }
    OCL_LOG("context created");
    if (err != CL_SUCCESS || !g->ctx) { release_all(g); free(g); return NULL; }
    g->queue = g->api.CreateCommandQueue(g->ctx, dev, 0, &err);
    OCL_LOG("queue created");
    if (err != CL_SUCCESS || !g->queue) { release_all(g); free(g); return NULL; }

    size_t src_len = strlen(OCL_SRC);
    const char *src = OCL_SRC;
    g->prog = g->api.CreateProgramWithSource(g->ctx, 1, &src, &src_len, &err);
    if (err != CL_SUCCESS || !g->prog) { release_all(g); free(g); return NULL; }
    if (g->api.BuildProgram(g->prog, 1, &dev, NULL, NULL, NULL) != CL_SUCCESS) {
        char log[4096] = { 0 };
        g->api.GetProgramBuildInfo(g->prog, dev, CL_PROGRAM_BUILD_LOG, sizeof log, log, NULL);
        log[sizeof log - 1] = 0;
        fprintf(stderr, "OpenCL: build failed:\n%s\n", log);
        release_all(g); free(g); return NULL;
    }
    g->kernel = g->api.CreateKernel(g->prog, "rt_main", &err);
    if (err != CL_SUCCESS || !g->kernel) {
        fprintf(stderr, "OpenCL: kernel creation failed (%d)\n", err);
        release_all(g); free(g); return NULL;
    }
    struct { const char *name; clh *dst; } kernels[] = {
        { "rt_logavg", &g->k_logavg }, { "rt_post", &g->k_post },
        { "rt_upscale", &g->k_upscale }, { "rt_cas", &g->k_cas },
        { "rt_tiles", &g->k_tiles },
    };
    for (size_t i = 0; i < sizeof kernels / sizeof kernels[0]; i++) {
        *kernels[i].dst = g->api.CreateKernel(g->prog, kernels[i].name, &err);
        if (err != CL_SUCCESS || !*kernels[i].dst) {
            fprintf(stderr, "OpenCL: %s creation failed (%d)\n", kernels[i].name, err);
            release_all(g); free(g); return NULL;
        }
    }
    OCL_LOG("kernel created");

    g->d_out = make_buffer(g, CL_MEM_READ_WRITE, (size_t)OCL_MAX_W * OCL_MAX_H * 16);
    g->d_sph = make_buffer(g, CL_MEM_READ_ONLY, OCL_MAX_SPHERES * 16);
    g->d_sph_mat = make_buffer(g, CL_MEM_READ_ONLY, OCL_MAX_SPHERES * 16);
    g->d_sph_emi = make_buffer(g, CL_MEM_READ_ONLY, OCL_MAX_SPHERES * 16);
    g->d_sph_texA = make_buffer(g, CL_MEM_READ_ONLY, OCL_MAX_SPHERES * 16);
    g->d_sph_texB = make_buffer(g, CL_MEM_READ_ONLY, OCL_MAX_SPHERES * 16);
    g->d_box_min = make_buffer(g, CL_MEM_READ_ONLY, OCL_MAX_BOXES * 16);
    g->d_box_max = make_buffer(g, CL_MEM_READ_ONLY, OCL_MAX_BOXES * 16);
    g->d_box_mat = make_buffer(g, CL_MEM_READ_ONLY, OCL_MAX_BOXES * 16);
    g->d_box_emi = make_buffer(g, CL_MEM_READ_ONLY, OCL_MAX_BOXES * 16);
    g->d_box_texA = make_buffer(g, CL_MEM_READ_ONLY, OCL_MAX_BOXES * 16);
    g->d_box_texB = make_buffer(g, CL_MEM_READ_ONLY, OCL_MAX_BOXES * 16);
    g->d_plane_pos = make_buffer(g, CL_MEM_READ_ONLY, OCL_MAX_PLANES * 16);
    g->d_plane_mat = make_buffer(g, CL_MEM_READ_ONLY, OCL_MAX_PLANES * 16);
    g->d_plane_emi = make_buffer(g, CL_MEM_READ_ONLY, OCL_MAX_PLANES * 16);
    g->d_plane_texA = make_buffer(g, CL_MEM_READ_ONLY, OCL_MAX_PLANES * 16);
    g->d_plane_texB = make_buffer(g, CL_MEM_READ_ONLY, OCL_MAX_PLANES * 16);
    g->d_cyl_b = make_buffer(g, CL_MEM_READ_ONLY, OCL_MAX_CYLS * 16);
    g->d_cyl_h = make_buffer(g, CL_MEM_READ_ONLY, OCL_MAX_CYLS * 16);
    g->d_cyl_mat = make_buffer(g, CL_MEM_READ_ONLY, OCL_MAX_CYLS * 16);
    g->d_cyl_emi = make_buffer(g, CL_MEM_READ_ONLY, OCL_MAX_CYLS * 16);
    g->d_cyl_texA = make_buffer(g, CL_MEM_READ_ONLY, OCL_MAX_CYLS * 16);
    g->d_cyl_texB = make_buffer(g, CL_MEM_READ_ONLY, OCL_MAX_CYLS * 16);
    g->d_lpos = make_buffer(g, CL_MEM_READ_ONLY, OCL_MAX_LIGHTS * 16);
    g->d_lcol = make_buffer(g, CL_MEM_READ_ONLY, OCL_MAX_LIGHTS * 16);
    g->d_lrad = make_buffer(g, CL_MEM_READ_ONLY, OCL_MAX_LIGHTS * 16);
    g->d_grid_a = make_buffer(g, CL_MEM_READ_ONLY, 16);
    g->d_grid_dims = make_buffer(g, CL_MEM_READ_ONLY, 16);
    g->d_grid_off = make_buffer(g, CL_MEM_READ_ONLY, (OCL_GRID_CELLS + 1) * 4);
    g->d_grid_tri = make_buffer(g, CL_MEM_READ_ONLY, (size_t)OCL_GRID_TRI_CAP * 4);
    g->d_rgb_in = make_buffer(g, CL_MEM_READ_WRITE, (size_t)OCL_MAX_W * OCL_MAX_H * 4);
    g->d_present = make_buffer(g, CL_MEM_READ_WRITE, (size_t)OCL_MAX_W * OCL_MAX_H * 4);
    g->d_cas = make_buffer(g, CL_MEM_READ_WRITE, (size_t)OCL_MAX_W * OCL_MAX_H * 4);
    g->d_logpart = make_buffer(g, CL_MEM_READ_WRITE, OCL_LOG_GROUPS * 4);
    g->d_frame = make_buffer(g, CL_MEM_READ_ONLY, FRAME_FLOATS * 4);
    g->d_tilebuf = make_buffer(g, CL_MEM_READ_ONLY, (size_t)OCL_MAX_TILES * 2 * sizeof(cl_int));
    if (!g->d_out || !g->d_sph || !g->d_sph_mat || !g->d_sph_emi || !g->d_sph_texA
        || !g->d_sph_texB || !g->d_box_min || !g->d_box_max || !g->d_box_mat
        || !g->d_box_emi || !g->d_box_texA || !g->d_box_texB || !g->d_cyl_b
        || !g->d_cyl_h || !g->d_cyl_mat || !g->d_cyl_emi || !g->d_cyl_texA
        || !g->d_cyl_texB || !g->d_plane_pos || !g->d_plane_mat || !g->d_plane_emi
        || !g->d_plane_texA || !g->d_plane_texB || !g->d_lpos || !g->d_lcol
        || !g->d_lrad || !g->d_grid_a || !g->d_grid_dims || !g->d_grid_off
        || !g->d_grid_tri || !g->d_rgb_in || !g->d_present || !g->d_cas
        || !g->d_logpart || !g->d_tilebuf || !g->d_frame) {
        fprintf(stderr, "OpenCL: buffer allocation failed\n");
        release_all(g); free(g); return NULL;
    }

    /* Bind every trace-kernel argument ONCE: SetKernelArg costs ~3 ms per
     * call on 51-arg kernels with the Gen9 driver, so the frame loop must
     * never call it. Per-frame scalars flow through d_frame (async write);
     * geometry buffers hold stable handles even before their contents are
     * uploaded by ocl_upload_scene. */
    {
        struct OclApi *api = &g->api;
        clh *bufs[] = {
            &g->d_sph, &g->d_sph_mat, &g->d_sph_emi, &g->d_sph_texA, &g->d_sph_texB,
            &g->d_box_min, &g->d_box_max, &g->d_box_mat, &g->d_box_emi,
            &g->d_box_texA, &g->d_box_texB,
            &g->d_cyl_b, &g->d_cyl_h, &g->d_cyl_mat, &g->d_cyl_emi,
            &g->d_cyl_texA, &g->d_cyl_texB,
            &g->d_plane_pos, &g->d_plane_mat, &g->d_plane_emi,
            &g->d_plane_texA, &g->d_plane_texB,
            &g->d_lpos, &g->d_lcol, &g->d_lrad,
        };
        struct { clh kernel; int has_tiles; } ks[] = {
            { g->kernel, 0 }, { g->k_tiles, 1 },
        };
        for (size_t k = 0; k < sizeof ks / sizeof ks[0]; k++) {
            cl_uint arg = 0;
            int ok = 1;
            ok &= api->SetKernelArg(ks[k].kernel, arg++, sizeof(clh), &g->d_out) == CL_SUCCESS;
            ok &= api->SetKernelArg(ks[k].kernel, arg++, sizeof(clh), &g->d_frame) == CL_SUCCESS;
            for (size_t i = 0; i < sizeof bufs / sizeof bufs[0]; i++)
                ok &= api->SetKernelArg(ks[k].kernel, arg++, sizeof(clh), bufs[i]) == CL_SUCCESS;
            ok &= api->SetKernelArg(ks[k].kernel, arg++, sizeof(clh), &g->d_tris) == CL_SUCCESS;
            ok &= api->SetKernelArg(ks[k].kernel, arg++, sizeof(clh), &g->d_grid_a) == CL_SUCCESS;
            ok &= api->SetKernelArg(ks[k].kernel, arg++, sizeof(clh), &g->d_grid_dims) == CL_SUCCESS;
            ok &= api->SetKernelArg(ks[k].kernel, arg++, sizeof(clh), &g->d_grid_off) == CL_SUCCESS;
            ok &= api->SetKernelArg(ks[k].kernel, arg++, sizeof(clh), &g->d_grid_tri) == CL_SUCCESS;
            if (ks[k].has_tiles) {
                clh d_tilebuf = g->d_tilebuf;
                cl_int tile_arg = 16; /* RT_HYBRID_TILE */
                ok &= api->SetKernelArg(ks[k].kernel, arg++, sizeof(clh), &d_tilebuf) == CL_SUCCESS;
                ok &= api->SetKernelArg(ks[k].kernel, arg++, sizeof(cl_int), &tile_arg) == CL_SUCCESS;
            }
            if (!ok) {
                fprintf(stderr, "OpenCL: kernel arg binding failed\n");
                release_all(g); free(g); return NULL;
            }
        }
    }
    fprintf(stderr, "OpenCL: renderer ready\n");
    return g;
}

void Ocl_Destroy(OclRenderer *g) {
    release_all(g);
    free(g);
}

static int upload_triangles(OclRenderer *g, const Scene *s) {
    if (g->tris_uploaded) return 1;
    int total = 0;
    for (int i = 0; i < s->count; i++) {
        if (s->objects[i].shape == RT_MESH) {
            total += s->meshes[s->objects[i].geometry.mesh.mesh].triangle_count;
        }
    }
    g->tri_count = total;
    g->tris_uploaded = 1;
    if (total <= 0) return 1;
    float *data = (float *)malloc((size_t)total * 9 * sizeof(float));
    if (!data) return 0;
    int tri = 0;
    for (int i = 0; i < s->count; i++) {
        if (s->objects[i].shape != RT_MESH) continue;
        const RtMesh *m = &s->meshes[s->objects[i].geometry.mesh.mesh];
        for (int t = 0; t < m->triangle_count; t++, tri++) {
            V3 v0 = m->vertices[m->triangles[t][0]];
            V3 v1 = m->vertices[m->triangles[t][1]];
            V3 v2 = m->vertices[m->triangles[t][2]];
            float *row = data + (size_t)tri * 9;
            row[0] = v0.x; row[1] = v0.y; row[2] = v0.z;
            row[3] = v1.x - v0.x; row[4] = v1.y - v0.y; row[5] = v1.z - v0.z;
            row[6] = v2.x - v0.x; row[7] = v2.y - v0.y; row[8] = v2.z - v0.z;
        }
    }
    cl_int err = 0;
    g->d_tris = g->api.CreateBuffer(g->ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    (size_t)total * 9 * sizeof(float), data, &err);
    if (err != CL_SUCCESS || !g->d_tris) { free(data); g->tri_count = 0; return 0; }
    /* d_tris was created after Ocl_Create bound the kernel args: rebind the
     * handle into both kernels once (a one-time SetKernelArg cost). */
    {
        clh tris = g->d_tris;
        g->api.SetKernelArg(g->kernel, 27, sizeof(clh), &tris);
        g->api.SetKernelArg(g->k_tiles, 27, sizeof(clh), &tris);
    }

    /* ---- build a uniform grid over the triangles (10x10x10 CSR) ---- */
    /* absolute vertex = v0 (+ e1) (+ e2); rows hold v0, e1, e2 */
#define OCL_TRI_VX(row, k, out) do { \
        (out)[0] = (row)[(k) * 3 + 0]; \
        (out)[1] = (row)[(k) * 3 + 1]; \
        (out)[2] = (row)[(k) * 3 + 2]; \
        if (k >= 1) { (out)[0] += (row)[0]; (out)[1] += (row)[1]; (out)[2] += (row)[2]; } \
        if (k == 2) { (out)[0] += (row)[3]; (out)[1] += (row)[4]; (out)[2] += (row)[5]; } \
    } while (0)

    V3 mn = { 1e30f, 1e30f, 1e30f }, mx = { -1e30f, -1e30f, -1e30f };
    for (int t = 0; t < total; t++) {
        const float *row = data + (size_t)t * 9;
        for (int k = 0; k < 3; k++) {
            float v[3];
            OCL_TRI_VX(row, k, v);
            mn.x = fminf(mn.x, v[0]); mn.y = fminf(mn.y, v[1]); mn.z = fminf(mn.z, v[2]);
            mx.x = fmaxf(mx.x, v[0]); mx.y = fmaxf(mx.y, v[1]); mx.z = fmaxf(mx.z, v[2]);
        }
    }
    for (int axis = 0; axis < 3; axis++) {
        float *lo = axis == 0 ? &mn.x : (axis == 1 ? &mn.y : &mn.z);
        float *hi = axis == 0 ? &mx.x : (axis == 1 ? &mx.y : &mx.z);
        *lo -= 1e-3f; *hi += 1e-3f;
        if (*hi - *lo < 1e-4f) *hi = *lo + 1e-4f;
    }
    int gx = 10, gy = 10, gz = 10;
    float cell = fmaxf((mx.x - mn.x) / gx, fmaxf((mx.y - mn.y) / gy, (mx.z - mn.z) / gz));
    g->h_grid_a[0] = mn.x; g->h_grid_a[1] = mn.y; g->h_grid_a[2] = mn.z;
    g->h_grid_a[3] = cell;
    g->h_grid_dims[0] = (unsigned)gx; g->h_grid_dims[1] = (unsigned)gy;
    g->h_grid_dims[2] = (unsigned)gz; g->h_grid_dims[3] = 0u;

    int ncell = gx * gy * gz;
    unsigned *cnt = (unsigned *)calloc((size_t)ncell, sizeof(unsigned));
    if (!cnt) { free(data); g->tri_count = 0; return 0; }
    for (int t = 0; t < total; t++) {
        const float *row = data + (size_t)t * 9;
        float vmin[3], vmax[3] = { 0 };
        OCL_TRI_VX(row, 0, vmin);
        vmax[0] = vmin[0]; vmax[1] = vmin[1]; vmax[2] = vmin[2];
        for (int k = 1; k < 3; k++) {
            float v[3];
            OCL_TRI_VX(row, k, v);
            vmin[0] = fminf(vmin[0], v[0]); vmax[0] = fmaxf(vmax[0], v[0]);
            vmin[1] = fminf(vmin[1], v[1]); vmax[1] = fmaxf(vmax[1], v[1]);
            vmin[2] = fminf(vmin[2], v[2]); vmax[2] = fmaxf(vmax[2], v[2]);
        }
        int x0 = (int)((vmin[0] - mn.x) / cell); int x1 = (int)((vmax[0] - mn.x) / cell);
        int y0 = (int)((vmin[1] - mn.y) / cell); int y1 = (int)((vmax[1] - mn.y) / cell);
        int z0 = (int)((vmin[2] - mn.z) / cell); int z1 = (int)((vmax[2] - mn.z) / cell);
        x0 = x0 < 0 ? 0 : x0; y0 = y0 < 0 ? 0 : y0; z0 = z0 < 0 ? 0 : z0;
        x1 = x1 >= gx ? gx - 1 : x1; y1 = y1 >= gy ? gy - 1 : y1; z1 = z1 >= gz ? gz - 1 : z1;
        for (int z = z0; z <= z1; z++)
            for (int y = y0; y <= y1; y++)
                for (int x = x0; x <= x1; x++)
                    cnt[z * gy * gx + y * gx + x]++;
    }
    g->h_grid_off[0] = 0;
    for (int c = 0; c < ncell; c++) g->h_grid_off[c + 1] = g->h_grid_off[c] + cnt[c];
    g->h_grid_tri_len = (int)g->h_grid_off[ncell];
    if (g->h_grid_tri_len > OCL_GRID_TRI_CAP) {
        fprintf(stderr, "OpenCL: grid overflow (%d refs > %d)\n",
                g->h_grid_tri_len, OCL_GRID_TRI_CAP);
        free(cnt); free(data); g->tri_count = 0; return 0;
    }
    g->h_grid_tri = (unsigned *)malloc((size_t)(g->h_grid_tri_len > 0 ? g->h_grid_tri_len : 1) * sizeof(unsigned));
    if (!g->h_grid_tri) { free(cnt); free(data); g->tri_count = 0; return 0; }
    unsigned *fill = (unsigned *)malloc((size_t)ncell * sizeof(unsigned));
    if (!fill) { free(cnt); free(g->h_grid_tri); g->h_grid_tri = NULL; free(data); g->tri_count = 0; return 0; }
    for (int c = 0; c < ncell; c++) fill[c] = g->h_grid_off[c];
    for (int t = 0; t < total; t++) {
        const float *row = data + (size_t)t * 9;
        float vmin[3], vmax[3] = { 0 };
        OCL_TRI_VX(row, 0, vmin);
        vmax[0] = vmin[0]; vmax[1] = vmin[1]; vmax[2] = vmin[2];
        for (int k = 1; k < 3; k++) {
            float v[3];
            OCL_TRI_VX(row, k, v);
            vmin[0] = fminf(vmin[0], v[0]); vmax[0] = fmaxf(vmax[0], v[0]);
            vmin[1] = fminf(vmin[1], v[1]); vmax[1] = fmaxf(vmax[1], v[1]);
            vmin[2] = fminf(vmin[2], v[2]); vmax[2] = fmaxf(vmax[2], v[2]);
        }
        int x0 = (int)((vmin[0] - mn.x) / cell); int x1 = (int)((vmax[0] - mn.x) / cell);
        int y0 = (int)((vmin[1] - mn.y) / cell); int y1 = (int)((vmax[1] - mn.y) / cell);
        int z0 = (int)((vmin[2] - mn.z) / cell); int z1 = (int)((vmax[2] - mn.z) / cell);
        x0 = x0 < 0 ? 0 : x0; y0 = y0 < 0 ? 0 : y0; z0 = z0 < 0 ? 0 : z0;
        x1 = x1 >= gx ? gx - 1 : x1; y1 = y1 >= gy ? gy - 1 : y1; z1 = z1 >= gz ? gz - 1 : z1;
        for (int z = z0; z <= z1; z++)
            for (int y = y0; y <= y1; y++)
                for (int x = x0; x <= x1; x++) {
                    int c = z * gy * gx + y * gx + x;
                    g->h_grid_tri[fill[c]++] = (unsigned)t;
                }
    }
    free(fill);
    free(cnt);
    free(data);
#undef OCL_TRI_VX

    /* upload grid once */
    float ga[4] = { g->h_grid_a[0], g->h_grid_a[1], g->h_grid_a[2], g->h_grid_a[3] };
    cl_uint gd[4] = { g->h_grid_dims[0], g->h_grid_dims[1], g->h_grid_dims[2], 0u };
    if (api_wb(g, g->d_grid_a, ga, 16) != CL_SUCCESS
        || api_wb(g, g->d_grid_dims, gd, 16) != CL_SUCCESS
        || api_wb(g, g->d_grid_off, g->h_grid_off, (OCL_GRID_CELLS + 1) * 4) != CL_SUCCESS
        || (g->h_grid_tri_len > 0
            && api_wb(g, g->d_grid_tri, g->h_grid_tri,
                      (size_t)g->h_grid_tri_len * 4) != CL_SUCCESS)) {
        g->tri_count = 0;
        return 0;
    }
    fprintf(stderr, "OpenCL: grid %dx%dx%d cell %.3f, %d tri refs for %d tris\n",
            gx, gy, gz, cell, g->h_grid_tri_len, total);
    return 1;
}

/* Shared scene marshal + upload + kernel-arg binding for rt_main (full
 * frame) and rt_tiles (hybrid tile batches). aspect is the camera aspect
 * ratio used for ray generation (presentation aspect for rt_main, film
 * aspect for rt_tiles). */
struct OclSceneInfo { int ns, nb, nc, npl, nl; int nargs; };

static int ocl_upload_scene(OclRenderer *g, const Scene *s, clh kernel,
                            int film_w, int film_h, float aspect, int spp,
                            int sample_base, int accumulating,
                            struct OclSceneInfo *info) {
    (void)kernel; /* args bound once in Ocl_Create */
    if (!upload_triangles(g, s)) { g->broken = 1; return 0; }

    struct OclApi *api = &g->api;
    static float sph[OCL_MAX_SPHERES * 4], sph_mat[OCL_MAX_SPHERES * 4], sph_emi[OCL_MAX_SPHERES * 4];
    static float sph_texA[OCL_MAX_SPHERES * 4], sph_texB[OCL_MAX_SPHERES * 4];
    static float box_min[OCL_MAX_BOXES * 4], box_max[OCL_MAX_BOXES * 4];
    static float box_mat[OCL_MAX_BOXES * 4], box_emi[OCL_MAX_BOXES * 4];
    static float box_texA[OCL_MAX_BOXES * 4], box_texB[OCL_MAX_BOXES * 4];
    static float plane_pos[OCL_MAX_PLANES * 4], plane_mat[OCL_MAX_PLANES * 4], plane_emi[OCL_MAX_PLANES * 4];
    static float plane_texA[OCL_MAX_PLANES * 4], plane_texB[OCL_MAX_PLANES * 4];
    static float cyl_b[OCL_MAX_CYLS * 4], cyl_h[OCL_MAX_CYLS * 4];
    static float cyl_mat[OCL_MAX_CYLS * 4], cyl_emi[OCL_MAX_CYLS * 4];
    static float cyl_texA[OCL_MAX_CYLS * 4], cyl_texB[OCL_MAX_CYLS * 4];
    static float lpos[OCL_MAX_LIGHTS * 4], lcol[OCL_MAX_LIGHTS * 4], lrad[OCL_MAX_LIGHTS * 4];

    int ns = 0, nb = 0, nc = 0, npl = 0;
    for (int i = 0; i < s->count; i++) {
        const Object *o = &s->objects[i];
        float *texA_dst = NULL, *texB_dst = NULL;
        int slot = -1;
        if (o->shape == RT_SPHERE && ns < OCL_MAX_SPHERES) { slot = ns; texA_dst = sph_texA; texB_dst = sph_texB; }
        else if (o->shape == RT_BOX && nb < OCL_MAX_BOXES) { slot = nb; texA_dst = box_texA; texB_dst = box_texB; }
        else if (o->shape == RT_CYLINDER && nc < OCL_MAX_CYLS) { slot = nc; texA_dst = cyl_texA; texB_dst = cyl_texB; }
        else if (o->shape == RT_PLANE && npl < OCL_MAX_PLANES) { slot = npl; texA_dst = plane_texA; texB_dst = plane_texB; }
        if (slot >= 0) {
            texA_dst[slot * 4 + 0] = (float)o->material.texture;
            texA_dst[slot * 4 + 1] = o->material.texture_scale;
            texA_dst[slot * 4 + 2] = o->material.texture_strength;
            texA_dst[slot * 4 + 3] = 0.0f;
            texB_dst[slot * 4 + 0] = o->material.texture_color.x;
            texB_dst[slot * 4 + 1] = o->material.texture_color.y;
            texB_dst[slot * 4 + 2] = o->material.texture_color.z;
            texB_dst[slot * 4 + 3] = 0.0f;
        }
        if (o->shape == RT_SPHERE && ns < OCL_MAX_SPHERES) {
            sph[ns * 4 + 0] = o->geometry.sphere.center.x;
            sph[ns * 4 + 1] = o->geometry.sphere.center.y;
            sph[ns * 4 + 2] = o->geometry.sphere.center.z;
            sph[ns * 4 + 3] = o->geometry.sphere.radius;
            sph_mat[ns * 4 + 0] = o->material.albedo.x;
            sph_mat[ns * 4 + 1] = o->material.albedo.y;
            sph_mat[ns * 4 + 2] = o->material.albedo.z;
            sph_mat[ns * 4 + 3] = o->material.reflection;
            sph_emi[ns * 4 + 0] = o->material.emission.x;
            sph_emi[ns * 4 + 1] = o->material.emission.y;
            sph_emi[ns * 4 + 2] = o->material.emission.z;
            sph_emi[ns * 4 + 3] = o->material.shininess;
            ns++;
        } else if (o->shape == RT_BOX && nb < OCL_MAX_BOXES) {
            box_min[nb * 4 + 0] = o->geometry.box.min.x;
            box_min[nb * 4 + 1] = o->geometry.box.min.y;
            box_min[nb * 4 + 2] = o->geometry.box.min.z;
            box_max[nb * 4 + 0] = o->geometry.box.max.x;
            box_max[nb * 4 + 1] = o->geometry.box.max.y;
            box_max[nb * 4 + 2] = o->geometry.box.max.z;
            box_mat[nb * 4 + 0] = o->material.albedo.x;
            box_mat[nb * 4 + 1] = o->material.albedo.y;
            box_mat[nb * 4 + 2] = o->material.albedo.z;
            box_mat[nb * 4 + 3] = o->material.reflection;
            box_emi[nb * 4 + 0] = o->material.emission.x;
            box_emi[nb * 4 + 1] = o->material.emission.y;
            box_emi[nb * 4 + 2] = o->material.emission.z;
            box_emi[nb * 4 + 3] = o->material.shininess;
            nb++;
        } else if (o->shape == RT_CYLINDER && nc < OCL_MAX_CYLS) {
            cyl_b[nc * 4 + 0] = o->geometry.cylinder.base.x;
            cyl_b[nc * 4 + 1] = o->geometry.cylinder.base.y;
            cyl_b[nc * 4 + 2] = o->geometry.cylinder.base.z;
            cyl_b[nc * 4 + 3] = o->geometry.cylinder.radius;
            cyl_h[nc * 4 + 0] = o->geometry.cylinder.base.y + o->geometry.cylinder.height;
            cyl_h[nc * 4 + 1] = 0.0f;
            cyl_h[nc * 4 + 2] = 0.0f;
            cyl_h[nc * 4 + 3] = 0.0f;
            cyl_mat[nc * 4 + 0] = o->material.albedo.x;
            cyl_mat[nc * 4 + 1] = o->material.albedo.y;
            cyl_mat[nc * 4 + 2] = o->material.albedo.z;
            cyl_mat[nc * 4 + 3] = o->material.reflection;
            cyl_emi[nc * 4 + 0] = o->material.emission.x;
            cyl_emi[nc * 4 + 1] = o->material.emission.y;
            cyl_emi[nc * 4 + 2] = o->material.emission.z;
            cyl_emi[nc * 4 + 3] = o->material.shininess;
            nc++;
        } else if (o->shape == RT_PLANE && npl < OCL_MAX_PLANES) {
            /* plane_pos: .x = checker flag, .w = plane height */
            plane_pos[npl * 4 + 0] = (float)(o->geometry.plane.checker != 0);
            plane_pos[npl * 4 + 3] = o->geometry.plane.y;
            plane_mat[npl * 4 + 0] = o->material.albedo.x;
            plane_mat[npl * 4 + 1] = o->material.albedo.y;
            plane_mat[npl * 4 + 2] = o->material.albedo.z;
            plane_mat[npl * 4 + 3] = o->material.reflection;
            plane_emi[npl * 4 + 0] = o->material.emission.x;
            plane_emi[npl * 4 + 1] = o->material.emission.y;
            plane_emi[npl * 4 + 2] = o->material.emission.z;
            plane_emi[npl * 4 + 3] = o->material.shininess;
            npl++;
        }
    }
    int nl = s->light_count < OCL_MAX_LIGHTS ? s->light_count : OCL_MAX_LIGHTS;
    for (int i = 0; i < nl; i++) {
        V3 lc = rt_light_rgb(&s->lights[i]);
        lpos[i * 4 + 0] = s->lights[i].position.x;
        lpos[i * 4 + 1] = s->lights[i].position.y;
        lpos[i * 4 + 2] = s->lights[i].position.z;
        lpos[i * 4 + 3] = s->lights[i].power;
        lcol[i * 4 + 0] = lc.x;
        lcol[i * 4 + 1] = lc.y;
        lcol[i * 4 + 2] = lc.z;
        lcol[i * 4 + 3] = 0.0f;
        lrad[i * 4 + 0] = s->lights[i].radius;
        lrad[i * 4 + 1] = 0.0f;
        lrad[i * 4 + 2] = 0.0f;
        lrad[i * 4 + 3] = 0.0f;
    }
    /* Sky sun for sky_color(): direction and blackbody color of the scene's
     * primary light, exactly like the CPU fallback light struct. */
    RtPointLight sun = { s->light, s->light_color, s->light_power,
                         s->light_radius, 1, 0.0f };
    V3 sun_rgb = rt_light_rgb(&sun);
    V3 sdir = vnorm(vsub(s->light, s->camera));
    float sun_dir[4] = { sdir.x, sdir.y, sdir.z, 0.0f };
    float sun_col[4] = { sun_rgb.x, sun_rgb.y, sun_rgb.z, 0.0f };
    float fog[4] = { s->fog_density, s->fog_color.x, s->fog_color.y, s->fog_color.z };

    /* camera basis, mirroring the CPU tracer / GLSL renderer */
    V3 fwd = vnorm(vsub(s->target, s->camera));
    V3 right = vnorm(vcross(fwd, v3(0.0f, 1.0f, 0.0f)));
    V3 up = vcross(right, fwd);
    float tan_h = tanf(s->fov * 3.14159265f / 360.0f);
    float aspect_tan[2] = { aspect * tan_h, tan_h };
    float cam_pos[4] = { s->camera.x, s->camera.y, s->camera.z, 0.0f };
    float f4[4] = { fwd.x, fwd.y, fwd.z, 0.0f };
    float r4[4] = { right.x, right.y, right.z, 0.0f };
    float u4[4] = { up.x, up.y, up.z, 0.0f };
    int counts[4] = { ns, nb, nc, npl };

    /* bunny material (geometry is accelerated by the uniform grid) */
    float bmat[4] = { 0.9f, 0.88f, 0.85f, 0.0f }, bemi[4] = { 0, 0, 0, 0 };
    for (int i = 0; i < s->count; i++) {
        if (s->objects[i].shape != RT_MESH) continue;
        bmat[0] = s->objects[i].material.albedo.x;
        bmat[1] = s->objects[i].material.albedo.y;
        bmat[2] = s->objects[i].material.albedo.z;
        bmat[3] = s->objects[i].material.reflection;
        bemi[0] = s->objects[i].material.emission.x;
        bemi[1] = s->objects[i].material.emission.y;
        bemi[2] = s->objects[i].material.emission.z;
        bemi[3] = s->objects[i].material.shininess;
    }

    /* Static geometry/materials upload once per scene; only sphere centers and
     * lights move frame-to-frame (pushable balls). All writes are async: the
     * in-order queue runs them before any kernel enqueued afterwards, and
     * blocking CL_TRUE writes cost ~15 ms each on the Gen9 driver. */
    if (g->uploaded_scene != s) {
        struct { clh *mem; const void *data; size_t size; } uploads[] = {
            { &g->d_sph_mat, sph_mat, (size_t)ns * 16 },
            { &g->d_sph_emi, sph_emi, (size_t)ns * 16 },
            { &g->d_sph_texA, sph_texA, (size_t)ns * 16 },
            { &g->d_sph_texB, sph_texB, (size_t)ns * 16 },
            { &g->d_box_min, box_min, (size_t)nb * 16 },
            { &g->d_box_max, box_max, (size_t)nb * 16 },
            { &g->d_box_mat, box_mat, (size_t)nb * 16 },
            { &g->d_box_emi, box_emi, (size_t)nb * 16 },
            { &g->d_box_texA, box_texA, (size_t)nb * 16 },
            { &g->d_box_texB, box_texB, (size_t)nb * 16 },
            { &g->d_cyl_b, cyl_b, (size_t)nc * 16 },
            { &g->d_cyl_h, cyl_h, (size_t)nc * 16 },
            { &g->d_cyl_mat, cyl_mat, (size_t)nc * 16 },
            { &g->d_cyl_emi, cyl_emi, (size_t)nc * 16 },
            { &g->d_cyl_texA, cyl_texA, (size_t)nc * 16 },
            { &g->d_cyl_texB, cyl_texB, (size_t)nc * 16 },
            { &g->d_plane_pos, plane_pos, (size_t)npl * 16 },
            { &g->d_plane_mat, plane_mat, (size_t)npl * 16 },
            { &g->d_plane_emi, plane_emi, (size_t)npl * 16 },
            { &g->d_plane_texA, plane_texA, (size_t)npl * 16 },
            { &g->d_plane_texB, plane_texB, (size_t)npl * 16 },
        };
        for (size_t i = 0; i < sizeof uploads / sizeof uploads[0]; i++) {
            if (uploads[i].size == 0) continue;
            if (api->EnqueueWriteBuffer(g->queue, *uploads[i].mem, CL_FALSE, 0, uploads[i].size,
                                        uploads[i].data, 0, NULL, NULL) != CL_SUCCESS) {
                g->broken = 1;
                return 0;
            }
        }
        g->uploaded_scene = s;
    }
    struct { clh *mem; const void *data; size_t size; } dynamic[] = {
        { &g->d_sph, sph, (size_t)ns * 16 },
        { &g->d_lpos, lpos, (size_t)nl * 16 },
        { &g->d_lcol, lcol, (size_t)nl * 16 },
        { &g->d_lrad, lrad, (size_t)nl * 16 },
    };
    for (size_t i = 0; i < sizeof dynamic / sizeof dynamic[0]; i++) {
        if (dynamic[i].size == 0) continue;
        if (api->EnqueueWriteBuffer(g->queue, *dynamic[i].mem, CL_FALSE, 0, dynamic[i].size,
                                    dynamic[i].data, 0, NULL, NULL) != CL_SUCCESS) {
            g->broken = 1;
            return 0;
        }
    }

    /* Per-frame scalars go through one buffer write; kernel args were bound
     * once in Ocl_Create (SetKernelArg is pathologically slow on Gen9). */
    float frame[FRAME_FLOATS];
    for (int i = 0; i < 4; i++) {
        frame[F_CAM + i] = cam_pos[i];
        frame[F_FWD + i] = f4[i];
        frame[F_RIGHT + i] = r4[i];
        frame[F_UP + i] = u4[i];
        frame[F_FOG + i] = fog[i];
        frame[F_SUN_DIR + i] = sun_dir[i];
        frame[F_SUN_COL + i] = sun_col[i];
        frame[F_BMAT + i] = bmat[i];
        frame[F_BEMI + i] = bemi[i];
    }
    frame[F_ASPECT] = aspect_tan[0];
    frame[F_ASPECT + 1] = aspect_tan[1];
    frame[F_W] = (float)film_w;
    frame[F_H] = (float)film_h;
    frame[F_SPP] = (float)spp;
    frame[F_SAMPLE_BASE] = (float)sample_base;
    frame[F_ACCUM] = (float)accumulating;
    frame[F_COUNTS] = (float)ns;
    frame[F_COUNTS + 1] = (float)nb;
    frame[F_COUNTS + 2] = (float)nc;
    frame[F_COUNTS + 3] = (float)npl;
    frame[F_TRI] = (float)g->tri_count;
    frame[F_LIGHTS] = (float)nl;
    /* Blocking on purpose: the Gen9 driver silently drops this write when it
     * is enqueued non-blocking and followed directly by a kernel launch. The
     * Finish also flushes the dynamic geometry writes above. */
    if (api->EnqueueWriteBuffer(g->queue, g->d_frame, CL_TRUE, 0,
                                sizeof frame, frame, 0, NULL, NULL) != CL_SUCCESS) {
        g->broken = 1;
        return 0;
    }

    if (info) {
        info->ns = ns; info->nb = nb; info->nc = nc;
        info->npl = npl; info->nl = nl; info->nargs = 0;
    }
    return 1;
}

int Ocl_GlInterop(const OclRenderer *g) {
    return g && g->d_gl != NULL;
}

int Ocl_AttachGlTexture(OclRenderer *g, unsigned tex_id, int width, int height) {
    if (!g || g->broken || !tex_id || width <= 0 || height <= 0) return 0;
    if (!g->api.CreateFromGLTexture || !g->api.EnqueueAcquireGLObjects ||
        !g->api.EnqueueReleaseGLObjects)
        return 0;
    if (width > OCL_MAX_W || height > OCL_MAX_H) return 0;
    cl_int err = 0;
    g->d_gl = g->api.CreateFromGLTexture(g->ctx, CL_MEM_WRITE_ONLY, CL_GL_TEXTURE_2D,
                                         0, tex_id, &err);
    if (err != CL_SUCCESS || !g->d_gl) {
        fprintf(stderr, "OpenCL: GL texture attach failed (%d), using readback\n", err);
        g->d_gl = NULL;
        return 0;
    }
    g->k_present = g->api.CreateKernel(g->prog, "rt_present", &err);
    if (err != CL_SUCCESS || !g->k_present) {
        fprintf(stderr, "OpenCL: rt_present creation failed (%d), using readback\n", err);
        g->api.ReleaseMemObject(g->d_gl);
        g->d_gl = NULL;
        return 0;
    }
    fprintf(stderr, "OpenCL: GL interop present active (%dx%d)\n", width, height);
    return 1;
}

static int ocl_render_internal(OclRenderer *g, const Scene *s, unsigned char *rgb,
               int width, int height, int internal_w, int internal_h,
               int spp, float cas, int accumulate) {
    if (!g || g->broken || !s || !rgb || width <= 0 || height <= 0) return 0;
    if (width > OCL_MAX_W || height > OCL_MAX_H) return 0;
    if (internal_w <= 0 || internal_h <= 0) { internal_w = width; internal_h = height; }
    if (internal_w > OCL_MAX_W || internal_h > OCL_MAX_H) return 0;
    if (spp < 1) spp = 1;
    if (!accumulate) g->accum_samples = 0;
    if (cas < 0.0f) cas = 0.0f;
    if (cas > 1.0f) cas = 1.0f;
    const int upscaling = internal_w != width || internal_h != height;
    struct OclApi *api = &g->api;

    struct OclSceneInfo info;
    /* ray generation uses the presentation aspect, exactly like before */
    const int sample_base = accumulate ? g->accum_samples : 0;
    if (!ocl_upload_scene(g, s, g->kernel, internal_w, internal_h,
                          (float)width / (float)height, spp,
                          sample_base, accumulate, &info)) return 0;
    if (accumulate && sample_base == 0) {
        static unsigned char *zeroes;
        static size_t zeroes_size;
        const size_t bytes = (size_t)internal_w * internal_h * 16;
        if (zeroes_size < bytes) {
            free(zeroes);
            zeroes = (unsigned char *)calloc(bytes, 1);
            zeroes_size = zeroes ? bytes : 0;
        }
        if (!zeroes || api_wb(g, g->d_out, zeroes, bytes) != CL_SUCCESS) {
            g->broken = 1;
            return 0;
        }
    }
    clh d_out = g->d_out;

    /* padded 2D workgroups keep the HD 620's SIMD lanes fully occupied even
     * when width/height are not multiples of the local size */
    const size_t local[2] = { 16, 16 };
    size_t global[2] = { ((size_t)internal_w + local[0] - 1) / local[0] * local[0],
                         ((size_t)internal_h + local[1] - 1) / local[1] * local[1] };
    LARGE_INTEGER qpc_freq, t0, t1, t2, t3;
    QueryPerformanceFrequency(&qpc_freq);
    QueryPerformanceCounter(&t0);
    int enq_ok = api->EnqueueNDRangeKernel(g->queue, g->kernel, 2, NULL, global, local,
                                           0, NULL, NULL) == CL_SUCCESS;
    int fin_ok = enq_ok && api->Finish(g->queue) == CL_SUCCESS;
    QueryPerformanceCounter(&t1);
    if (!enq_ok || !fin_ok) {
        g->broken = 1;
        return 0;
    }
    /* GPU post chain: log-avg reduction -> host exposure math (shared
     * adaptation state) -> bloom/Reinhard/sRGB kernel -> optional upscale+CAS.
     * Same math as rt_postprocess_hdr (postprocess_hdr/exposed_hdr). */
    const size_t inpix = (size_t)internal_w * internal_h;
    const size_t lglobal[1] = { OCL_LOG_GROUPS * OCL_LOG_LOCAL };
    const size_t llocal[1] = { OCL_LOG_LOCAL };
    cl_int npix_arg = (cl_int)inpix;
    clh d_logpart = g->d_logpart;
    clh k_l = g->k_logavg;
    int post_ok = 1;
    post_ok &= api->SetKernelArg(k_l, 0, sizeof(clh), &d_out) == CL_SUCCESS;
    post_ok &= api->SetKernelArg(k_l, 1, sizeof(cl_int), &npix_arg) == CL_SUCCESS;
    post_ok &= api->SetKernelArg(k_l, 2, OCL_LOG_LOCAL * sizeof(float), NULL) == CL_SUCCESS;
    post_ok &= api->SetKernelArg(k_l, 3, sizeof(clh), &d_logpart) == CL_SUCCESS;
    post_ok &= api->EnqueueNDRangeKernel(g->queue, k_l, 1, NULL, lglobal, llocal,
                                         0, NULL, NULL) == CL_SUCCESS;
    if (!post_ok) { g->broken = 1; return 0; }
    float partials[OCL_LOG_GROUPS];
    if (api->EnqueueReadBuffer(g->queue, g->d_logpart, CL_TRUE, 0,
                               sizeof partials, partials, 0, NULL, NULL) != CL_SUCCESS) {
        g->broken = 1;
        return 0;
    }
    double log_sum = 0.0;
    for (int i = 0; i < OCL_LOG_GROUPS; i++) log_sum += (double)partials[i];
    float exposure = rt_exposure_from_logavg(log_sum, (int)inpix);

    clh k_p = g->k_post, d_rgb_in = g->d_rgb_in;
    post_ok = 1;
    post_ok &= api->SetKernelArg(k_p, 0, sizeof(clh), &d_out) == CL_SUCCESS;
    post_ok &= api->SetKernelArg(k_p, 1, sizeof(clh), &d_rgb_in) == CL_SUCCESS;
    post_ok &= api->SetKernelArg(k_p, 2, sizeof(int), &internal_w) == CL_SUCCESS;
    post_ok &= api->SetKernelArg(k_p, 3, sizeof(int), &internal_h) == CL_SUCCESS;
    post_ok &= api->SetKernelArg(k_p, 4, sizeof(float), &exposure) == CL_SUCCESS;
    post_ok &= api->EnqueueNDRangeKernel(g->queue, k_p, 2, NULL, global, local,
                                         0, NULL, NULL) == CL_SUCCESS;
    if (!post_ok) { g->broken = 1; return 0; }

    clh read_buf = d_rgb_in;
    if (upscaling) {
        clh k_u = g->k_upscale, d_present = g->d_present;
        post_ok = 1;
        post_ok &= api->SetKernelArg(k_u, 0, sizeof(clh), &d_rgb_in) == CL_SUCCESS;
        post_ok &= api->SetKernelArg(k_u, 1, sizeof(clh), &d_present) == CL_SUCCESS;
        post_ok &= api->SetKernelArg(k_u, 2, sizeof(int), &internal_w) == CL_SUCCESS;
        post_ok &= api->SetKernelArg(k_u, 3, sizeof(int), &internal_h) == CL_SUCCESS;
        post_ok &= api->SetKernelArg(k_u, 4, sizeof(int), &width) == CL_SUCCESS;
        post_ok &= api->SetKernelArg(k_u, 5, sizeof(int), &height) == CL_SUCCESS;
        const size_t gout[2] = { ((size_t)width + local[0] - 1) / local[0] * local[0],
                                 ((size_t)height + local[1] - 1) / local[1] * local[1] };
        post_ok &= api->EnqueueNDRangeKernel(g->queue, k_u, 2, NULL, gout, local,
                                             0, NULL, NULL) == CL_SUCCESS;
        read_buf = d_present;
        if (post_ok && cas > 0.0f) {
            clh k_c = g->k_cas, d_cas = g->d_cas;
            float cas_arg = cas;
            post_ok &= api->SetKernelArg(k_c, 0, sizeof(clh), &d_present) == CL_SUCCESS;
            post_ok &= api->SetKernelArg(k_c, 1, sizeof(clh), &d_cas) == CL_SUCCESS;
            post_ok &= api->SetKernelArg(k_c, 2, sizeof(int), &width) == CL_SUCCESS;
            post_ok &= api->SetKernelArg(k_c, 3, sizeof(int), &height) == CL_SUCCESS;
            post_ok &= api->SetKernelArg(k_c, 4, sizeof(float), &cas_arg) == CL_SUCCESS;
            post_ok &= api->EnqueueNDRangeKernel(g->queue, k_c, 2, NULL, gout, local,
                                                 0, NULL, NULL) == CL_SUCCESS;
            read_buf = d_cas;
        }
        if (!post_ok) { g->broken = 1; return 0; }
    }
    if (g->d_gl) {
        /* Zero-copy present: copy the final frame into the GL-shared texture
         * while it is acquired; the viewer draws it directly. */
        int gl_ok = 1;
        clh k_pr = g->k_present, d_gl = g->d_gl;
        gl_ok &= api->SetKernelArg(k_pr, 0, sizeof(clh), &read_buf) == CL_SUCCESS;
        gl_ok &= api->SetKernelArg(k_pr, 1, sizeof(clh), &d_gl) == CL_SUCCESS;
        gl_ok &= api->SetKernelArg(k_pr, 2, sizeof(int), &width) == CL_SUCCESS;
        gl_ok &= api->SetKernelArg(k_pr, 3, sizeof(int), &height) == CL_SUCCESS;
        gl_ok &= api->EnqueueAcquireGLObjects(g->queue, 1, &d_gl, 0, NULL, NULL) == CL_SUCCESS;
        const size_t pglobal[2] = { ((size_t)width + local[0] - 1) / local[0] * local[0],
                                    ((size_t)height + local[1] - 1) / local[1] * local[1] };
        gl_ok &= api->EnqueueNDRangeKernel(g->queue, k_pr, 2, NULL, pglobal, local,
                                           0, NULL, NULL) == CL_SUCCESS;
        gl_ok &= api->EnqueueReleaseGLObjects(g->queue, 1, &d_gl, 0, NULL, NULL) == CL_SUCCESS;
        if (!gl_ok) { g->broken = 1; return 0; }
    }
    if (api->Finish(g->queue) != CL_SUCCESS) { g->broken = 1; return 0; }
    QueryPerformanceCounter(&t2);

    if (g->d_gl) {
        /* readback skipped: the frame lives in the GL texture now */
        QueryPerformanceCounter(&t3);
    } else {
    /* readback: 8-bit sRGB rgba at presentation size, repack to rgb888 */
    const size_t outpix = (size_t)width * height;
    static unsigned char *rgba = NULL;
    static size_t rcap = 0;
    if (rcap < outpix) {
        free(rgba);
        rgba = (unsigned char *)malloc(outpix * 4);
        if (!rgba) { rcap = 0; g->broken = 1; return 0; }
        rcap = outpix;
    }
    if (api->EnqueueReadBuffer(g->queue, read_buf, CL_TRUE, 0,
                               outpix * 4, rgba, 0, NULL, NULL) != CL_SUCCESS) {
        g->broken = 1;
        return 0;
    }
    for (size_t i = 0; i < outpix; i++) {
        rgb[i * 3 + 0] = rgba[i * 4 + 0];
        rgb[i * 3 + 1] = rgba[i * 4 + 1];
        rgb[i * 3 + 2] = rgba[i * 4 + 2];
    }
    QueryPerformanceCounter(&t3);
    }

    static int dump_once = 0;
    if (!dump_once) {
        dump_once = 1;
        unsigned char px[3] = { 0, 0, 0 };
        if (g->d_gl) {
            unsigned char px4[4] = { 0, 0, 0, 0 };
            api->EnqueueReadBuffer(g->queue, read_buf, CL_TRUE, 0, 4, px4, 0, NULL, NULL);
            px[0] = px4[0]; px[1] = px4[1]; px[2] = px4[2];
        } else {
            px[0] = rgb[0]; px[1] = rgb[1]; px[2] = rgb[2];
        }
        fprintf(stderr, "OpenCL: first px %d %d %d | ns=%d nb=%d nc=%d npl=%d nl=%d tri=%d spp=%d\n",
                px[0], px[1], px[2], info.ns, info.nb, info.nc, info.npl,
                info.nl, g->tri_count, spp);
    }
    if (++g->frames % 30 == 0 && getenv("RT_PROFILE")) {
        double kms = (double)(t1.QuadPart - t0.QuadPart) * 1000.0 / qpc_freq.QuadPart;
        double pms = (double)(t2.QuadPart - t1.QuadPart) * 1000.0 / qpc_freq.QuadPart;
        double rms = (double)(t3.QuadPart - t2.QuadPart) * 1000.0 / qpc_freq.QuadPart;
        double total = (double)(t3.QuadPart - t0.QuadPart) * 1000.0 / qpc_freq.QuadPart;
        fprintf(stderr, "OpenCL: frame %d kernel %.1f post %.1f readback %.1f | total %.1f ms (%.1f fps) res %dx%d\n",
                g->frames, kms, pms, rms, total, total > 0.0 ? 1000.0 / total : 0.0,
                internal_w, internal_h);
        fflush(stderr);
    }
    if (accumulate) g->accum_samples += spp;
    return 1;
}

int Ocl_Render(OclRenderer *g, const Scene *s, unsigned char *rgb,
               int width, int height, int internal_w, int internal_h,
               int spp, float cas) {
    return ocl_render_internal(g, s, rgb, width, height, internal_w, internal_h,
                               spp, cas, 0);
}

int Ocl_RenderAccum(OclRenderer *g, const Scene *s, unsigned char *rgb,
                    int width, int height, int internal_w, int internal_h,
                    int spp, float cas, int reset) {
    if (reset) g->accum_samples = 0;
    return ocl_render_internal(g, s, rgb, width, height, internal_w, internal_h,
                               spp, cas, 1);
}

/* Hybrid mode, phase 1: enqueue an rt_tiles kernel over the given tile batch
 * without blocking the CPU (async tile-list upload + async kernel). The CPU
 * keeps tracing its own tiles meanwhile; Ocl_ReadTiles does the join. */
int Ocl_TraceTiles(OclRenderer *g, const Scene *s, int width, int height,
                   int spp, const int *tiles_xy, int tile_count, int tile) {
    if (!g || g->broken || !s || !tiles_xy || tile_count <= 0) return 0;
    if (tile_count > OCL_MAX_TILES) return 0;
    if (width <= 0 || width > OCL_MAX_W || height <= 0 || height > OCL_MAX_H) return 0;
    if (spp < 1) spp = 1;
    if (tile != 16) return 0; /* tile arg is bound once at Ocl_Create */
    g->tile_size = tile;

    struct OclSceneInfo info;
    /* film aspect: matches the CPU ray generator pixel for pixel */
    if (!ocl_upload_scene(g, s, g->k_tiles, width, height,
                          (float)width / (float)height, spp, 0, 0, &info)) return 0;
    struct OclApi *api = &g->api;

    if (api->EnqueueWriteBuffer(g->queue, g->d_tilebuf, CL_FALSE, 0,
                                (size_t)tile_count * 2 * sizeof(cl_int),
                                tiles_xy, 0, NULL, NULL) != CL_SUCCESS) {
        fprintf(stderr, "OpenCL: rt_tiles tile-list upload failed\n");
        g->broken = 1;
        return 0;
    }
    /* Workgroup-size query returns 0 on this driver (Gen9 OpenCL 1.2 quirk),
     * so probe locally: halve until the enqueue is accepted. */
    const size_t local = (size_t)tile * (size_t)tile;
    cl_int e3 = -999;
    for (size_t try_local = local; try_local >= 1; try_local /= 2) {
        const size_t try_global = try_local * (size_t)tile_count;
        e3 = api->EnqueueNDRangeKernel(g->queue, g->k_tiles, 1, NULL,
                                       &try_global, &try_local, 0, NULL, NULL);
        if (e3 == CL_SUCCESS) break;
        fprintf(stderr, "OpenCL: rt_tiles enqueue local=%zu -> %d\n", try_local, e3);
    }
    if (e3 != CL_SUCCESS) {
        fprintf(stderr, "OpenCL: rt_tiles enqueue failed (%d)\n", e3);
        g->broken = 1;
        return 0;
    }
    return 1; /* not finished yet: CPU workers keep going */
}

/* Hybrid mode, phase 2: join (wait for the GPU), then copy the traced tiles
 * from the HDR film buffer into hdr. *kernel_ms (if non-NULL) receives the
 * wall time of the join, i.e. the GPU tail of this batch. */
int Ocl_ReadTiles(OclRenderer *g, V3 *hdr, int width, int height,
                  const int *tiles_xy, int tile_count, double *kernel_ms) {
    if (!g || g->broken || !hdr || !tiles_xy || tile_count <= 0) return 0;
    if (width <= 0 || width > OCL_MAX_W || height <= 0 || height > OCL_MAX_H) return 0;
    const int ts = g->tile_size > 0 ? g->tile_size : 16;
    struct OclApi *api = &g->api;

    LARGE_INTEGER qpc_freq, t0, t1;
    QueryPerformanceFrequency(&qpc_freq);
    QueryPerformanceCounter(&t0);
    if (api->Finish(g->queue) != CL_SUCCESS) { g->broken = 1; return 0; }
    QueryPerformanceCounter(&t1);
    if (kernel_ms)
        *kernel_ms = (double)(t1.QuadPart - t0.QuadPart) * 1000.0 / qpc_freq.QuadPart;

    /* UMA: read the whole film back in one shot (fast), then scatter the
     * tiles this batch owns into the shared HDR buffer. */
    const size_t npix = (size_t)width * height;
    static float *film = NULL; /* float4 staging */
    static size_t fcap = 0;
    if (fcap < npix) {
        free(film);
        film = (float *)malloc(npix * 4 * sizeof(float));
        if (!film) { fcap = 0; g->broken = 1; return 0; }
        fcap = npix;
    }
    if (api->EnqueueReadBuffer(g->queue, g->d_out, CL_TRUE, 0,
                               npix * 4 * sizeof(float), film, 0, NULL, NULL) != CL_SUCCESS) {
        g->broken = 1;
        return 0;
    }
    for (int t = 0; t < tile_count; t++) {
        const int x0 = tiles_xy[t * 2 + 0];
        const int y0 = tiles_xy[t * 2 + 1];
        const int x1 = x0 + ts < width ? x0 + ts : width;
        const int y1 = y0 + ts < height ? y0 + ts : height;
        for (int y = y0; y < y1; y++) {
            const float *row = film + (size_t)y * width * 4;
            for (int x = x0; x < x1; x++)
                hdr[y * width + x] = v3(row[x * 4 + 0], row[x * 4 + 1], row[x * 4 + 2]);
        }
    }
    return 1;
}
