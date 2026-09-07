# Architecture and API Reference

## 1. System Overview

Verlet Lab contains four mostly independent systems:

1. A deterministic analytic CPU ray tracer.
2. A standalone material/BRDF research library.
3. 2D and 3D Verlet physics experiments.
4. A visual-only Clay editor shell.

The systems share the repository but do not share runtime state. This is an
intentional safety boundary. The viewer may use the tracer; the UI does not.

```text
world_viewer.c
    -> tracer_scene.c
    -> tracer.c / tracer.h
    -> CPU framebuffer
    -> raylib texture/window

render_world.c
    -> tracer_scene.c
    -> tracer.c / tracer.h
    -> BMP output

material_test.c
    -> src/materials/*.c
    -> deterministic scalar checks

main3d.c
    -> verlet3d.h
    -> raylib / rlgl

ui.c
    -> Clay + raylib renderer
    -> mock editor panels only
```

## 2. Directory Ownership

### `src/raytracer`

`tracer.h` defines the public ray-tracer data model. `tracer.c` implements
analytic intersections, acceleration, lighting, recursive radiance, scalar
rendering, and the AVX2 exact-pixel path. `tracer_scene.c` owns scene assembly.
`render_world.c` is a headless BMP driver. `world_viewer.c` owns input, camera
movement, dynamic ball pushing, texture upload, and the interactive window.

### `src/materials`

This is a standalone shader library. It has its own vector type and API so its
PBR experiments do not accidentally change the established tracer output.
The intended future migration is explicit: add surface data to the tracer,
then replace legacy Blinn-Phong parameters with material evaluation.

### `src/physics2d`

`main.c` is the original 2D Verlet demonstration. It has its own world, body,
constraint, rigid-shape, and raylib drawing code.

### `src/physics3d`

`verlet3d.h` contains the 3D simulation core and public types. `main3d.c` is
the raylib water/physics front-end. `fluid_fast.h` contains the fast fluid
support code used by the 3D experiment.

### `src/ui`

`ui.c` is a Clay v0.14 editor shell. It presents hierarchy, viewport, asset,
console, timeline, and inspector panels using mock state. It is not connected
to the tracer or physics simulation.

### `tests`

`tests/materials/material_test.c` checks shader math, Fresnel, refraction,
textures, dispatch, and all named presets. `tests/physics3d/` contains the
existing simulation smoke tests and determinism checks.

## 3. Ray-Tracer Data Model

### `V3`

Three-component single-precision vector. Inline helpers include `vadd`, `vsub`,
`vscale`, `vmul`, `vdot`, `vcross`, `vnorm`, and `vlen`.

### `Material`

Legacy tracer material used by the currently integrated renderer:

```c
typedef struct {
    V3 albedo;
    float reflection;
    float shininess;
    V3 emission;
} Material;
```

`albedo` controls diffuse color, `reflection` controls recursive/environment
reflection weight, `shininess` controls the legacy Blinn-Phong exponent, and
`emission` adds radiance independent of direct lighting.

### `Object`

An object has a `Shape`, `Material`, and a tagged geometry union:

- `RT_BOX`: axis-aligned min/max bounds.
- `RT_SPHERE`: center and radius.
- `RT_CYLINDER`: vertical closed cylinder with base, radius, and height.
- `RT_PLANE`: horizontal plane with optional world-space checker pattern.

### `Scene`

Contains up to `RT_MAX_OBJECTS` objects, a static BVH, up to
`RT_MAX_LIGHTS` point lights, camera/target/FOV, and compatibility fields for
the original single-light path. The interactive scene also stores object IDs
for its red and blue pushable balls.

### `Hit`

```c
typedef struct { float t; V3 normal; int object; } Hit;
```

`t` is the ray distance, `normal` is the analytic surface normal, and `object`
is the scene object index.

## 4. Ray-Tracer Functions

### Geometry and acceleration

- `rt_intersect(object, origin, direction, t_max, hit)`: dispatches to one
  primitive intersection routine.
- `rt_trace(scene, origin, direction, t_max, hit)`: nearest-hit query using
  the BVH plus infinite planes.
- `rt_build_accel(scene)`: builds the static median-split BVH after scene
  assembly or dynamic-object movement.
- `rt_occluded(scene, origin, light)`: finite shadow query.

### Radiance and rendering

- `rt_radiance(scene, origin, direction, depth)`: reference recursive radiance
  with five soft-shadow samples and bounded reflection.
- `rt_render(scene, rgb, width, height, samples)`: reference deterministic
  framebuffer renderer.
- `rt_render_rows(...)`: renders a row interval for external worker systems.
- `rt_render_blocked(...)`: legacy sparse-grid reconstruction path. It remains
  available for experiments but is not used by the current viewer.
- `rt_render_exact(...)`: scalar one-ray-per-output-pixel renderer.
- `rt_render_exact_fast(...)`: AVX2 packet renderer used by the native viewer.

## 5. Interactive World

`scene_small()` creates the compact 1x1-performance scene. It contains a
plaza, two buildings, two mirrors, two emissive spheres, and three point
lights. `world_viewer.c` starts at the plaza center. Walking into either light
ball pushes it horizontally; its associated point light follows the sphere and
the acceleration structure is rebuilt after movement.

`scene_world()` remains the larger 70-object diorama for headless rendering.
`scene_block()` remains the small verification scene.

## 6. Material Library API

The material library's public header is `src/materials/materials.h`.

Core types:

- `RtmVec3`: standalone material-library vector.
- `ShaderMaterial`: model, base color, emission, IOR, roughness, metallic,
  transmission, conductor eta/k, clearcoat, sheen, subsurface, and thickness.
- `RtmResponse`: reflection/transmission/emission event result.
- `RtmPreset`: stable enum for the 41 named materials.

Core functions:

- Vector functions: `rtm_v3`, `rtm_add`, `rtm_sub`, `rtm_scale`, `rtm_mul`,
  `rtm_lerp`, `rtm_dot`, `rtm_length`, `rtm_normalize`.
- BRDF math: `rtm_fresnel_dielectric`, `rtm_fresnel_conductor`,
  `rtm_ggx_distribution`, `rtm_smith_visibility`, `rtm_lambert`.
- Models: `rtm_eval_reflection`, `rtm_eval_transmission`,
  `rtm_eval_layered`, `rtm_eval_sheen`, `rtm_eval_subsurface`,
  `rtm_eval_bsdf`, `rtm_surface_response`.
- Transport: `rtm_refract`.
- Textures: `rtm_hash3`, `rtm_noise3`, `rtm_fbm`, checker, stripes, wood,
  marble, brick, rust, and fabric functions.
- Presets: `rtm_material_preset`, `rtm_preset_name`, `rtm_preset_count`.

## 7. Build Targets

The PowerShell scripts under `scripts/` are the canonical Windows commands:

- `build_raytracer.ps1`: builds `build/render_world.exe`.
- `build_viewer.ps1`: builds the native AVX2/OpenMP viewer.
- `build_materials.ps1`: builds the material test executable.
- `build_physics_tests.ps1`: builds the 3D physics tests.
- `run_checks.ps1`: builds and runs the deterministic checks.

The scripts use paths relative to the project root and do not modify global Git
configuration. Raylib remains an external sibling dependency at
`C:/Users/kono/Desktop/DEV/C/raylib-5.5` in the current environment.

## 8. Ownership and Extension Rules

- Add new shapes to `Shape`, the geometry union, intersection dispatch, bounds,
  and documentation together.
- Add lights through scene builders; keep light counts bounded.
- Rebuild the BVH after any object geometry movement.
- Keep reference and interactive render paths separately named.
- Add a material preset by parameterizing an existing model before creating a
  new shader branch.
- Preserve deterministic output in headless paths.
- Do not wire `ui.c` mock state to runtime systems without an explicit design
  change and tests.
