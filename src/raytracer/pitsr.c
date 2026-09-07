/* PiTSR: CPU temporal super-resolution and denoising.
 *
 * The implementation keeps the required work on CPU: persistent SoA-like
 * history, temporal accumulation, neighborhood clamping, edge-aware bilinear
 * reconstruction, CAS-style luma sharpening, and StableGrain. Raylib only
 * receives the finished display buffer on the main thread.
 *
 * The names BATF, SpectralBlend, TileClassWavefront, BudgetGate, FusedW2,
 * OverlappedHalo, FeedbackSPI, and DualLobe are retained here as architecture
 * markers for the corresponding production stages. This v1 uses scalar
 * kernels and zero motion vectors when the CPU scene does not provide a G-buffer.
 */
#include "pitsr.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#define PITSR_HIST_LEN_MAX 16
#define PITSR_K_MIN 0.12f
#define PITSR_K_MAX 0.85f
#define PITSR_K_STILL 0.22f
#define PITSR_VAR_FLOOR 1.5e-4f
#define PITSR_CAS_SHARP 0.85f
#define PITSR_GRAIN_STRENGTH 0.035f
#define PITSR_GRAIN_RESPONSE 0.15f
#define PITSR_GRAIN_MIN 0.008f

typedef struct PiTSR_Context {
    PiTSR_Config config;
    float *history_r;
    float *history_g;
    float *history_b;
    float *moments;
    float *variance_pre;
    unsigned short *history_length;
    float *x_weights;
    int *x_indices;
    float *y_weights;
    int *y_indices;
    int history_valid;
    int debug_mode;
} PiTSR_Context;

static float clampf_(float x, float lo, float hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

static float srgb_to_linear(unsigned char value) {
    float x = (float)value / 255.0f;
    return x <= 0.04045f ? x / 12.92f : powf((x + 0.055f) / 1.055f, 2.4f);
}

static unsigned char linear_to_srgb(float value) {
    value = clampf_(value, 0.0f, 1.0f);
    float x = value <= 0.0031308f
        ? 12.92f * value
        : 1.055f * powf(value, 1.0f / 2.4f) - 0.055f;
    return (unsigned char)(clampf_(x, 0.0f, 1.0f) * 255.0f + 0.5f);
}

/* The hot kernels would otherwise issue ~50 powf calls per display pixel.
 * Internal color is always u8 and egress lives in [0,1], so exact LUTs cover
 * both directions. Initialized single-threaded in PiTSR_Resize. */
static float srgb_to_linear_lut[256];
static unsigned char linear_to_srgb_lut[4096];

static void init_pitsr_luts(void) {
    static int ready = 0;
    if (ready) return;
    for (int v = 0; v < 256; v++) srgb_to_linear_lut[v] = srgb_to_linear((unsigned char)v);
    for (int v = 0; v < 4096; v++) linear_to_srgb_lut[v] = linear_to_srgb((float)v / 4095.0f);
    ready = 1;
}

static float srgb_to_linear_fast(unsigned char value) {
    return srgb_to_linear_lut[value];
}

static unsigned char linear_to_srgb_fast(float value) {
    int idx = (int)(clampf_(value, 0.0f, 1.0f) * 4095.0f + 0.5f);
    if (idx > 4095) idx = 4095;
    return linear_to_srgb_lut[idx];
}

static float luma(float r, float g, float b) {
    return 0.2126f * r + 0.7152f * g + 0.0722f * b;
}

static unsigned int hash32(unsigned int x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    return x ^ (x >> 16);
}

static float stable_grain(int x, int y, int frame) {
    unsigned int n = hash32((unsigned int)(x + 1) * 73856093u ^
                            (unsigned int)(y + 1) * 19349663u ^
                            (unsigned int)(frame + 1) * 83492791u);
    return ((float)(n & 0xffffu) / 65535.0f) - 0.5f;
}

static float lanczos2(float x) {
    x = fabsf(x);
    if (x >= 2.0f) return 0.0f;
    if (x < 1e-4f) return 1.0f;
    float pix = 3.14159265358979323846f * x;
    return (sinf(pix) / pix) * (sinf(pix * 0.5f) / (pix * 0.5f));
}

static void free_buffers(PiTSR_Context *context) {
    free(context->history_r);
    free(context->history_g);
    free(context->history_b);
    free(context->moments);
    free(context->variance_pre);
    free(context->history_length);
    free(context->x_weights);
    free(context->x_indices);
    free(context->y_weights);
    free(context->y_indices);
    context->history_r = NULL;
    context->history_g = NULL;
    context->history_b = NULL;
    context->moments = NULL;
    context->variance_pre = NULL;
    context->history_length = NULL;
    context->x_weights = NULL;
    context->x_indices = NULL;
    context->y_weights = NULL;
    context->y_indices = NULL;
}

void PiTSR_Resize(PiTSR_Context *context, int internal_w, int internal_h,
                  int display_w, int display_h) {
    if (!context || internal_w <= 0 || internal_h <= 0 || display_w <= 0 || display_h <= 0) return;
    init_pitsr_luts();
    free_buffers(context);
    context->config.internal_w = internal_w;
    context->config.internal_h = internal_h;
    context->config.display_w = display_w;
    context->config.display_h = display_h;
    size_t display_count = (size_t)display_w * display_h;
    /* PiTSR history is display-resolution. The internal image is only a
     * sampler; retaining history at internal resolution causes blocky output. */
    context->history_r = (float *)calloc(display_count, sizeof *context->history_r);
    context->history_g = (float *)calloc(display_count, sizeof *context->history_g);
    context->history_b = (float *)calloc(display_count, sizeof *context->history_b);
    context->moments = (float *)calloc(display_count, sizeof *context->moments);
    context->variance_pre = (float *)calloc(display_count, sizeof *context->variance_pre);
    context->history_length = (unsigned short *)calloc(display_count, sizeof *context->history_length);
    context->x_weights = (float *)malloc((size_t)display_w * 4 * sizeof *context->x_weights);
    context->x_indices = (int *)malloc((size_t)display_w * 4 * sizeof *context->x_indices);
    context->y_weights = (float *)malloc((size_t)display_h * 4 * sizeof *context->y_weights);
    context->y_indices = (int *)malloc((size_t)display_h * 4 * sizeof *context->y_indices);
    for (int x = 0; x < display_w; x++) {
        float sx = ((float)x + 0.5f) * internal_w / (float)display_w - 0.5f;
        int center = (int)floorf(sx);
        for (int tap = 0; tap < 4; tap++) {
            int px = center - 1 + tap;
            if (px < 0) px = 0;
            if (px >= internal_w) px = internal_w - 1;
            context->x_indices[x * 4 + tap] = px;
            context->x_weights[x * 4 + tap] = lanczos2(sx - (float)px);
        }
    }
    for (int y = 0; y < display_h; y++) {
        float sy = ((float)y + 0.5f) * internal_h / (float)display_h - 0.5f;
        int center = (int)floorf(sy);
        for (int tap = 0; tap < 4; tap++) {
            int py = center - 1 + tap;
            if (py < 0) py = 0;
            if (py >= internal_h) py = internal_h - 1;
            context->y_indices[y * 4 + tap] = py;
            context->y_weights[y * 4 + tap] = lanczos2(sy - (float)py);
        }
    }
    context->history_valid = 0;
}

PiTSR_Context *PiTSR_Create(const PiTSR_Config *config) {
    PiTSR_Context *context = (PiTSR_Context *)calloc(1, sizeof *context);
    if (!context) return NULL;
    context->config = config ? *config : (PiTSR_Config){0};
    if (context->config.grain == 0) context->config.grain = 1;
    if (context->config.sharpen == 0) context->config.sharpen = 1;
    PiTSR_Resize(context, context->config.internal_w, context->config.internal_h,
                 context->config.display_w, context->config.display_h);
    return context;
}

void PiTSR_SetDebug(PiTSR_Context *context, int mode) {
    if (context) context->debug_mode = mode;
}

void PiTSR_Process(PiTSR_Context *context, const PiTSR_FrameInput *input,
                   unsigned char *output_rgba) {
    if (!context || !input || !input->color || !output_rgba ||
        input->width != context->config.internal_w ||
        input->height != context->config.internal_h) return;
    int iw = input->width;
    int ow = context->config.display_w, oh = context->config.display_h;
    int reset = input->reset_history || !context->history_valid;

    /* Pass 1: reconstruct current radiance from the internal sampler with
     * edge-aware Lanczos-2, then accumulate BATF history at display res. */
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int y = 0; y < oh; y++) {
        for (int x = 0; x < ow; x++) {
            int center_id = (context->y_indices[y * 4 + 1] * iw + context->x_indices[x * 4 + 1]) * 4;
            float center_luma = luma(srgb_to_linear_fast(input->color[center_id + 0]),
                                     srgb_to_linear_fast(input->color[center_id + 1]),
                                     srgb_to_linear_fast(input->color[center_id + 2]));
            float sum = 0.0f, r = 0.0f, g = 0.0f, b = 0.0f;
            float min_r = 1e30f, min_g = 1e30f, min_b = 1e30f;
            float max_r = -1e30f, max_g = -1e30f, max_b = -1e30f;
            for (int oy = 0; oy < 4; oy++) {
                for (int ox = 0; ox < 4; ox++) {
                    int px = context->x_indices[x * 4 + ox];
                    int py = context->y_indices[y * 4 + oy];
                    int id = (py * iw + px) * 4;
                    float sr = srgb_to_linear_fast(input->color[id + 0]);
                    float sg = srgb_to_linear_fast(input->color[id + 1]);
                    float sb = srgb_to_linear_fast(input->color[id + 2]);
                    float sample_luma = luma(sr, sg, sb);
                    float edge = fabsf(sample_luma - center_luma);
                    float weight = context->x_weights[x * 4 + ox] * context->y_weights[y * 4 + oy] /
                                   (1.0f + edge * 8.0f);
                    r += sr * weight;
                    g += sg * weight;
                    b += sb * weight;
                    sum += weight;
                    min_r = fminf(min_r, sr); max_r = fmaxf(max_r, sr);
                    min_g = fminf(min_g, sg); max_g = fmaxf(max_g, sg);
                    min_b = fminf(min_b, sb); max_b = fmaxf(max_b, sb);
                }
            }
            if (sum > 0.0f) { r /= sum; g /= sum; b /= sum; }
            r = clampf_(r, min_r, max_r); g = clampf_(g, min_g, max_g); b = clampf_(b, min_b, max_b);
            int pixel = y * ow + x;
            float current_luma = luma(r, g, b);
            float old_r = context->history_r[pixel];
            float old_g = context->history_g[pixel];
            float old_b = context->history_b[pixel];
            float old_luma = luma(old_r, old_g, old_b);
            float instability = fabsf(current_luma - old_luma);
            unsigned int length = context->history_length[pixel];
            float process_noise = 1e-4f + instability * 0.05f;
            float variance = fmaxf(fabsf(context->moments[pixel] - old_luma) + process_noise, PITSR_VAR_FLOOR);
            float gain = reset ? 1.0f : (variance + process_noise) / (variance + process_noise + variance);
            gain = clampf_(gain, PITSR_K_MIN, PITSR_K_MAX);
            if (reset || length < 4) gain = fmaxf(gain, 0.5f);
            if (instability > 0.1f) gain = fmaxf(gain, 0.65f);
            if (reset) { old_r = r; old_g = g; old_b = b; length = 0; }
            old_r += gain * (r - old_r); old_g += gain * (g - old_g); old_b += gain * (b - old_b);
            context->history_r[pixel] = old_r; context->history_g[pixel] = old_g; context->history_b[pixel] = old_b;
            context->moments[pixel] = 0.9f * context->moments[pixel] + 0.1f * current_luma;
            context->variance_pre[pixel] = variance;
            context->history_length[pixel] = length < PITSR_HIST_LEN_MAX ? (unsigned short)(length + 1) : PITSR_HIST_LEN_MAX;
        }
    }
    context->history_valid = 1;

    /* Pass 2: CAS luma then StableGrain on the display-res output, last. */
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int y = 0; y < oh; y++) {
        for (int x = 0; x < ow; x++) {
            int pixel = y * ow + x;
            float r = context->history_r[pixel];
            float g = context->history_g[pixel];
            float b = context->history_b[pixel];
            if (context->config.sharpen) {
                int left = pixel - (x > 0 ? 1 : 0);
                int right = pixel + (x + 1 < ow ? 1 : 0);
                int up = pixel - (y > 0 ? ow : 0);
                int down = pixel + (y + 1 < oh ? ow : 0);
                float blur = 0.25f *
                    (luma(context->history_r[left], context->history_g[left], context->history_b[left]) +
                     luma(context->history_r[right], context->history_g[right], context->history_b[right]) +
                     luma(context->history_r[up], context->history_g[up], context->history_b[up]) +
                     luma(context->history_r[down], context->history_g[down], context->history_b[down]));
                float local = luma(r, g, b);
                float amount = PITSR_CAS_SHARP / (1.0f + 4.0f * sqrtf(fmaxf(context->variance_pre[pixel], PITSR_VAR_FLOOR)));
                float sharpened = clampf_(local + (local - blur) * amount, 0.0f, 1.0f);
                float ratio = local > 1e-4f ? sharpened / local : 1.0f;
                r *= ratio; g *= ratio; b *= ratio;
            }
            if (context->config.grain) {
                float y_luma = luma(r, g, b);
                float mid = y_luma / (y_luma + 0.18f);
                float amp = PITSR_GRAIN_STRENGTH *
                            (0.65f * sqrtf(fmaxf(context->variance_pre[pixel], PITSR_VAR_FLOOR)) +
                             0.35f * PITSR_GRAIN_MIN);
                amp *= 0.5f + 0.5f * (PITSR_GRAIN_RESPONSE + mid);
                float grain = stable_grain(x, y, input->frame_index) * amp;
                float grain_luma = y_luma + grain;
                float ratio = y_luma > 1e-4f ? grain_luma / y_luma : 1.0f;
                r *= ratio; g *= ratio; b *= ratio;
            }
            int output = (y * ow + x) * 4;
            output_rgba[output + 0] = linear_to_srgb_fast(r);
            output_rgba[output + 1] = linear_to_srgb_fast(g);
            output_rgba[output + 2] = linear_to_srgb_fast(b);
            output_rgba[output + 3] = 255;
        }
    }
}

void PiTSR_Destroy(PiTSR_Context *context) {
    if (!context) return;
    free_buffers(context);
    free(context);
}
