#ifndef VERLET_INPUT_H
#define VERLET_INPUT_H

#include "raylib.h"

#define VL_INPUT_MAX_GAMEPADS 4

typedef enum {
    VL_PAD_A = 0,
    VL_PAD_B,
    VL_PAD_X,
    VL_PAD_Y,
    VL_PAD_LEFT_BUMPER,
    VL_PAD_RIGHT_BUMPER,
    VL_PAD_BACK,
    VL_PAD_START,
    VL_PAD_GUIDE,
    VL_PAD_LEFT_THUMB,
    VL_PAD_RIGHT_THUMB,
    VL_PAD_DPAD_UP,
    VL_PAD_DPAD_RIGHT,
    VL_PAD_DPAD_DOWN,
    VL_PAD_DPAD_LEFT,
    VL_PAD_COUNT
} VlPadButton;

typedef enum {
    VL_PAD_LEFT_X = 0,
    VL_PAD_LEFT_Y,
    VL_PAD_RIGHT_X,
    VL_PAD_RIGHT_Y,
    VL_PAD_LEFT_TRIGGER,
    VL_PAD_RIGHT_TRIGGER,
    VL_PAD_AXIS_COUNT
} VlPadAxis;

typedef struct {
    float left_deadzone;
    float right_deadzone;
    float trigger_deadzone;
    float response_curve;
} VlInputConfig;

typedef struct {
    int id;
    int available;
    const char *name;
    int axis_count;
    float left_x, left_y;
    float right_x, right_y;
    float left_trigger, right_trigger;
    unsigned char buttons[VL_PAD_COUNT];
    unsigned char pressed[VL_PAD_COUNT];
    unsigned char released[VL_PAD_COUNT];
} VlGamepadState;

typedef struct {
    VlInputConfig config;
    VlGamepadState gamepads[VL_INPUT_MAX_GAMEPADS];
    Vector2 mouse_position;
    Vector2 mouse_delta;
    Vector2 mouse_wheel;
    unsigned char mouse_buttons[8];
    unsigned char mouse_pressed[8];
    unsigned char mouse_released[8];
    int last_key;
    int last_char;
} VlInput;

void vl_input_init(VlInput *input);
void vl_input_begin_frame(VlInput *input);
void vl_input_end_frame(VlInput *input);
void vl_input_set_config(VlInput *input, VlInputConfig config);

int vl_key_down(int key);
int vl_key_pressed(int key);
int vl_key_released(int key);
int vl_key_up(int key);
int vl_key_last(void);
int vl_char_last(void);

Vector2 vl_mouse_position(const VlInput *input);
Vector2 vl_mouse_delta(const VlInput *input);
Vector2 vl_mouse_wheel(const VlInput *input);
int vl_mouse_down(const VlInput *input, int button);
int vl_mouse_pressed(const VlInput *input, int button);
int vl_mouse_released(const VlInput *input, int button);
void vl_mouse_set_position(Vector2 position);
void vl_mouse_set_scale(float scale_x, float scale_y);

int vl_gamepad_available(int id);
const VlGamepadState *vl_gamepad(const VlInput *input, int id);
const char *vl_gamepad_name(int id);
int vl_gamepad_axis_count(int id);
float vl_gamepad_axis(const VlInput *input, int id, VlPadAxis axis);
float vl_gamepad_trigger(const VlInput *input, int id, VlPadAxis axis);
int vl_gamepad_down(const VlInput *input, int id, VlPadButton button);
int vl_gamepad_pressed(const VlInput *input, int id, VlPadButton button);
int vl_gamepad_released(const VlInput *input, int id, VlPadButton button);
void vl_gamepad_rumble(int id, float left_motor, float right_motor, float seconds);

/* Raylib exposes standardized controls and rumble. Battery and hardware-level
   capability telemetry are backend-dependent and intentionally not fabricated. */
int vl_gamepad_has_rumble(void);
int vl_gamepad_has_battery_telemetry(void);

#endif
