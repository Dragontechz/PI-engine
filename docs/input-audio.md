# Input and Audio Libraries

## Separation

`src/input/input.h/.c` and `src/audio/audio.h/.c` are independent libraries.
Neither library includes or stores state from the other. Both currently use
raylib as the platform backend, but callers depend only on the `vl_*` APIs.

## Input

`VlInput` owns per-frame snapshots for keyboard, mouse, and up to four
gamepads. Call `vl_input_begin_frame()` after raylib's frame polling and use
the query functions during gameplay. The snapshot contains current, pressed,
and released states for mouse buttons and Xbox-compatible gamepad buttons.

Supported Xbox-style controls include A/B/X/Y, bumpers, back/start/guide,
thumb-clicks, D-pad, both sticks, and both analog triggers. Stick dead zones
and response curves are applied before values are exposed. `vl_gamepad_rumble`
maps directly to the backend's two motor vibration call.

`vl_gamepad_name`, `vl_gamepad_axis_count`, and capability functions expose
backend information. Raylib 5.5 does not expose controller battery telemetry
or a reliable hardware feature bitset, so `vl_gamepad_has_battery_telemetry()`
returns zero instead of pretending that data exists.

## Audio

`VlAudio` owns the device and a small bus mixer model. The backend remains
raylib, while the library adds:

- Master, music, SFX, UI, ambience, and custom buses.
- Bus volume and mute state.
- Sound loading, aliases, playback, pause/resume, pitch, volume, and pan.
- Streamed music playback, seeking, looping, pitch, volume, pan, and time.
- Generated sine-tone creation for tests and UI feedback.
- Distance/pan spatial playback helper.

The bus abstraction is a gain policy applied when playback starts or a handle
is configured. It is intentionally lightweight; a future mixer backend can
replace it without changing game-side calls.

## Build

```powershell
./scripts/build_input_audio.ps1
./build/input_test.exe
./build/audio_test.exe
```

The audio test treats an unavailable device as an environmental condition and
does not fail the source/API test. A real application should surface that
condition to its platform layer.
