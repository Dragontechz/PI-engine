#include "input.h"
#include <math.h>
#include <string.h>

static VlInput *active_input;
static float mouse_scale_x = 1.0f;
static float mouse_scale_y = 1.0f;

static float clamp01_(float x) { return x < 0.0f ? 0.0f : (x > 1.0f ? 1.0f : x); }

static float deadzone_axis(float value, float deadzone, float curve) {
    float sign = value < 0.0f ? -1.0f : 1.0f;
    float magnitude = fabsf(value);
    if (magnitude <= deadzone) return 0.0f;
    magnitude = (magnitude - deadzone) / (1.0f - deadzone);
    if (curve > 0.0f && curve != 1.0f) magnitude = powf(magnitude, curve);
    return sign * clamp01_(magnitude);
}

static int valid_button(int button) { return button >= 0 && button < 8; }
static int valid_pad(int id) { return id >= 0 && id < VL_INPUT_MAX_GAMEPADS; }

void vl_input_init(VlInput *input) {
    if (!input) return;
    memset(input, 0, sizeof *input);
    input->config.left_deadzone = 0.12f;
    input->config.right_deadzone = 0.12f;
    input->config.trigger_deadzone = 0.04f;
    input->config.response_curve = 1.0f;
    active_input = input;
}

void vl_input_set_config(VlInput *input, VlInputConfig config) {
    if (!input) return;
    input->config.left_deadzone = clamp01_(config.left_deadzone);
    input->config.right_deadzone = clamp01_(config.right_deadzone);
    input->config.trigger_deadzone = clamp01_(config.trigger_deadzone);
    input->config.response_curve = config.response_curve > 0.0f ? config.response_curve : 1.0f;
}

void vl_input_begin_frame(VlInput *input) {
    if (!input) return;
    active_input = input;
    input->mouse_position = GetMousePosition();
    input->mouse_delta = GetMouseDelta();
    input->mouse_delta.x *= mouse_scale_x;
    input->mouse_delta.y *= mouse_scale_y;
    input->mouse_wheel = GetMouseWheelMoveV();
    input->last_key = GetKeyPressed();
    input->last_char = GetCharPressed();
    for (int b = 0; b < 8; b++) {
        input->mouse_buttons[b] = IsMouseButtonDown(b);
        input->mouse_pressed[b] = IsMouseButtonPressed(b);
        input->mouse_released[b] = IsMouseButtonReleased(b);
    }
    for (int id = 0; id < VL_INPUT_MAX_GAMEPADS; id++) {
        VlGamepadState *pad = &input->gamepads[id];
        pad->id = id;
        pad->available = IsGamepadAvailable(id);
        pad->name = pad->available ? GetGamepadName(id) : "";
        pad->axis_count = pad->available ? GetGamepadAxisCount(id) : 0;
        if (!pad->available) {
            memset(pad->buttons, 0, sizeof pad->buttons);
            memset(pad->pressed, 0, sizeof pad->pressed);
            memset(pad->released, 0, sizeof pad->released);
            continue;
        }
        pad->left_x = deadzone_axis(GetGamepadAxisMovement(id, GAMEPAD_AXIS_LEFT_X), input->config.left_deadzone, input->config.response_curve);
        pad->left_y = deadzone_axis(GetGamepadAxisMovement(id, GAMEPAD_AXIS_LEFT_Y), input->config.left_deadzone, input->config.response_curve);
        pad->right_x = deadzone_axis(GetGamepadAxisMovement(id, GAMEPAD_AXIS_RIGHT_X), input->config.right_deadzone, input->config.response_curve);
        pad->right_y = deadzone_axis(GetGamepadAxisMovement(id, GAMEPAD_AXIS_RIGHT_Y), input->config.right_deadzone, input->config.response_curve);
        pad->left_trigger = clamp01_(deadzone_axis(GetGamepadAxisMovement(id, GAMEPAD_AXIS_LEFT_TRIGGER), input->config.trigger_deadzone, 1.0f));
        pad->right_trigger = clamp01_(deadzone_axis(GetGamepadAxisMovement(id, GAMEPAD_AXIS_RIGHT_TRIGGER), input->config.trigger_deadzone, 1.0f));
        for (int b = 0; b < VL_PAD_COUNT; b++) {
            int mapped = GAMEPAD_BUTTON_RIGHT_FACE_DOWN;
            if (b == VL_PAD_A) mapped = GAMEPAD_BUTTON_RIGHT_FACE_DOWN;
            else if (b == VL_PAD_B) mapped = GAMEPAD_BUTTON_RIGHT_FACE_RIGHT;
            else if (b == VL_PAD_X) mapped = GAMEPAD_BUTTON_RIGHT_FACE_LEFT;
            else if (b == VL_PAD_Y) mapped = GAMEPAD_BUTTON_RIGHT_FACE_UP;
            else if (b == VL_PAD_LEFT_THUMB) mapped = GAMEPAD_BUTTON_LEFT_THUMB;
            else if (b == VL_PAD_RIGHT_THUMB) mapped = GAMEPAD_BUTTON_RIGHT_THUMB;
            if (b == VL_PAD_GUIDE) mapped = GAMEPAD_BUTTON_MIDDLE;
            else if (b == VL_PAD_BACK) mapped = GAMEPAD_BUTTON_MIDDLE_LEFT;
            else if (b == VL_PAD_START) mapped = GAMEPAD_BUTTON_MIDDLE_RIGHT;
            else if (b >= VL_PAD_DPAD_UP && b <= VL_PAD_DPAD_LEFT) mapped = GAMEPAD_BUTTON_LEFT_FACE_UP + (b - VL_PAD_DPAD_UP);
            pad->buttons[b] = IsGamepadButtonDown(id, mapped);
            pad->pressed[b] = IsGamepadButtonPressed(id, mapped);
            pad->released[b] = IsGamepadButtonReleased(id, mapped);
        }
        pad->buttons[VL_PAD_LEFT_BUMPER] = IsGamepadButtonDown(id, GAMEPAD_BUTTON_LEFT_TRIGGER_1);
        pad->buttons[VL_PAD_RIGHT_BUMPER] = IsGamepadButtonDown(id, GAMEPAD_BUTTON_RIGHT_TRIGGER_1);
        pad->pressed[VL_PAD_LEFT_BUMPER] = IsGamepadButtonPressed(id, GAMEPAD_BUTTON_LEFT_TRIGGER_1);
        pad->pressed[VL_PAD_RIGHT_BUMPER] = IsGamepadButtonPressed(id, GAMEPAD_BUTTON_RIGHT_TRIGGER_1);
        pad->released[VL_PAD_LEFT_BUMPER] = IsGamepadButtonReleased(id, GAMEPAD_BUTTON_LEFT_TRIGGER_1);
        pad->released[VL_PAD_RIGHT_BUMPER] = IsGamepadButtonReleased(id, GAMEPAD_BUTTON_RIGHT_TRIGGER_1);
    }
}

void vl_input_end_frame(VlInput *input) { if (input) active_input = input; }
void vl_mouse_set_position(Vector2 position) { SetMousePosition((int)position.x, (int)position.y); }
void vl_mouse_set_scale(float x, float y) { mouse_scale_x = x; mouse_scale_y = y; }
int vl_key_down(int key) { return IsKeyDown(key); }
int vl_key_pressed(int key) { return IsKeyPressed(key); }
int vl_key_released(int key) { return IsKeyReleased(key); }
int vl_key_up(int key) { return IsKeyUp(key); }
int vl_key_last(void) { return active_input ? active_input->last_key : 0; }
int vl_char_last(void) { return active_input ? active_input->last_char : 0; }
Vector2 vl_mouse_position(const VlInput *i) { return i ? i->mouse_position : (Vector2){0}; }
Vector2 vl_mouse_delta(const VlInput *i) { return i ? i->mouse_delta : (Vector2){0}; }
Vector2 vl_mouse_wheel(const VlInput *i) { return i ? i->mouse_wheel : (Vector2){0}; }
int vl_mouse_down(const VlInput *i, int b) { return i && valid_button(b) && i->mouse_buttons[b]; }
int vl_mouse_pressed(const VlInput *i, int b) { return i && valid_button(b) && i->mouse_pressed[b]; }
int vl_mouse_released(const VlInput *i, int b) { return i && valid_button(b) && i->mouse_released[b]; }
int vl_gamepad_available(int id) { return valid_pad(id) && IsGamepadAvailable(id); }
const VlGamepadState *vl_gamepad(const VlInput *i, int id) { return i && valid_pad(id) ? &i->gamepads[id] : NULL; }
const char *vl_gamepad_name(int id) { return vl_gamepad_available(id) ? GetGamepadName(id) : ""; }
int vl_gamepad_axis_count(int id) { return vl_gamepad_available(id) ? GetGamepadAxisCount(id) : 0; }
float vl_gamepad_axis(const VlInput *i, int id, VlPadAxis a) { const VlGamepadState *p = vl_gamepad(i,id); if (!p || a < 0 || a >= VL_PAD_AXIS_COUNT) return 0.0f; return a == VL_PAD_LEFT_X ? p->left_x : a == VL_PAD_LEFT_Y ? p->left_y : a == VL_PAD_RIGHT_X ? p->right_x : a == VL_PAD_RIGHT_Y ? p->right_y : a == VL_PAD_LEFT_TRIGGER ? p->left_trigger : p->right_trigger; }
float vl_gamepad_trigger(const VlInput *i, int id, VlPadAxis a) { return vl_gamepad_axis(i,id,a); }
int vl_gamepad_down(const VlInput *i, int id, VlPadButton b) { const VlGamepadState *p=vl_gamepad(i,id); return p && b >= 0 && b < VL_PAD_COUNT && p->buttons[b]; }
int vl_gamepad_pressed(const VlInput *i, int id, VlPadButton b) { const VlGamepadState *p=vl_gamepad(i,id); return p && b >= 0 && b < VL_PAD_COUNT && p->pressed[b]; }
int vl_gamepad_released(const VlInput *i, int id, VlPadButton b) { const VlGamepadState *p=vl_gamepad(i,id); return p && b >= 0 && b < VL_PAD_COUNT && p->released[b]; }
void vl_gamepad_rumble(int id, float left, float right, float seconds) { if (vl_gamepad_available(id)) SetGamepadVibration(id, clamp01_(left), clamp01_(right), seconds > 0.0f ? seconds : 0.0f); }
int vl_gamepad_has_rumble(void) { return 1; }
int vl_gamepad_has_battery_telemetry(void) { return 0; }
