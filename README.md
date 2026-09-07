# Verlet Lab

CPU physics, ray tracing, materials, and experimental rendering tools written
in C. The project is intentionally split by responsibility so the interactive
viewer, deterministic headless renderer, physics experiments, and visual-only
editor can evolve independently.

## Quick Start

From `C/verlet` in Windows PowerShell:

```powershell
./scripts/build_raytracer.ps1
./scripts/build_viewer.ps1
./scripts/build_materials.ps1
./scripts/run_checks.ps1
```

Run the interactive viewer:

```powershell
./build/world_viewer.exe
```

Controls:

- `WASD`: move.
- `Q/E`: descend/ascend.
- `Shift`: sprint.
- Mouse: look.
- `R`: reset to the central plaza.
- `F1`: toggle help.
- `Escape`: release or capture the mouse.

The viewer is the exact-pixel CPU path: its 1280x720 framebuffer receives one
primary ray per output pixel. It uses AVX2 packets when built with native CPU
flags. The current i5-7300U benchmark is approximately 45 FPS for the compact
interactive scene.

## Layout

```text
src/raytracer/       CPU ray tracer, scenes, headless renderer, viewer
src/materials/       standalone BRDF/PBR shader library and presets
src/input/           keyboard, mouse, Xbox-compatible gamepad, and rumble API
src/audio/           independent audio device, buses, sounds, music, and tones
src/physics2d/       legacy 2D Verlet application
src/physics3d/       deterministic 3D Verlet core and raylib front-end
src/ui/              Clay editor shell, visual-only by design
tools/               image conversion helpers
tests/               physics and material checks
assets/              fonts and authored image sequences
docs/                architecture/API and math/optimization references
scripts/              reproducible PowerShell build/check commands
```

## Documentation

- `docs/architecture.md`: complete code map, data flow, public APIs, targets,
  ownership rules, and integration boundaries.
- `docs/math-and-optimization.md`: equations, intersection details, shading,
  BVH construction, AVX2 packets, memory layout, threading, and benchmarks.
- `docs/materials.md`: material-library design and the 41 documented presets.
- `docs/input-audio.md`: independent input/audio APIs, Xbox coverage, and
  backend capability limits.

## Important Boundaries

- `src/raytracer/` is deterministic and headless-verifiable.
- `src/materials/` is currently standalone and is not silently wired into the
  legacy tracer.
- `src/ui/ui.c` is visual-only mock editor state. It must not drive the tracer
  without an explicit integration change.
- `src/physics3d/main3d.c` is the 3D water/physics front-end; its renderer and
  simulation are separate from the CPU ray tracer.
- `src/input/` and `src/audio/` are independent libraries. They currently use
  raylib as a backend but do not depend on one another or on the viewer.
- Generated binaries, captures, logs, and render outputs are ignored by Git.
