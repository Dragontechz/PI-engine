# PiTSR CPU Upscaler

`src/raytracer/pitsr.c` implements the CPU-only `--mode upscale` path for the
world viewer. The ray-traced scene is rendered at half display resolution, then
PiTSR performs the following work without GL or shader filters:

- Persistent linear RGB history and fp32 luma moments.
- Camera-motion history reset because the current CPU scene has no motion-vector
  or depth G-buffer.
- BATF-style Kalman accumulation with luma instability process noise.
- Edge-aware reconstruction from the temporal history.
- Luma-only sharpening and deterministic StableGrain at final egress.
- Main-thread RGBA upload through the existing Raylib `UpdateTexture` call.

The current implementation uses scalar C11 kernels and a fallback zero-motion
model. It does not claim the full AVX2/F16C, depth-aware SVGF, or motion-vector
feature set until the renderer exports those G-buffer planes. No second GLFW
context is created and no worker touches Raylib or OpenGL.

Because the current CPU ray tracer does not provide depth or camera motion
vectors, the viewer resets PiTSR history on any camera translation or rotation.
This follows the no-MV fallback rule: a sharp noisy frame is preferable to
smearing history from a different surface. History resumes when the camera
stops. Motion-vector units are therefore not synthesized in this version.

Modes remain separate:

- `--mode upscale`: PiTSR CPU temporal super-resolution.
- `--mode adaptive`: full-resolution adaptive sample scheduler.
- `--mode native`: full-resolution uniform renderer.
