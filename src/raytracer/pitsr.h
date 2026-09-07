#ifndef PITSR_H
#define PITSR_H

#include <stddef.h>

typedef struct {
    const unsigned char *color;
    int width;
    int height;
    float jitter_x;
    float jitter_y;
    float exposure;
    int frame_index;
    int reset_history;
} PiTSR_FrameInput;

typedef struct {
    int internal_w;
    int internal_h;
    int display_w;
    int display_h;
    int threads;
    float time_budget_ms;
    int preset;
    int grain;
    int sharpen;
} PiTSR_Config;

typedef struct PiTSR_Context PiTSR_Context;

PiTSR_Context *PiTSR_Create(const PiTSR_Config *config);
void PiTSR_Resize(PiTSR_Context *context, int internal_w, int internal_h,
                  int display_w, int display_h);
void PiTSR_Process(PiTSR_Context *context, const PiTSR_FrameInput *input,
                   unsigned char *output_rgba);
void PiTSR_SetDebug(PiTSR_Context *context, int mode);
void PiTSR_Destroy(PiTSR_Context *context);

#endif
