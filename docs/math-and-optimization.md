# Math and Optimization Reference

## 1. Coordinate and Ray Conventions

All tracer vectors are single-precision `V3` values. A ray is:

```text
R(t) = O + tD
```

where `O` is the origin, `D` is normalized, and positive `t` travels forward.
The tracer rejects hits below `RT_EPSILON = 0.0005` to avoid self-intersection
acne.

The camera basis is:

```text
F = normalize(target - camera)
R = normalize(F x world_up)
U = R x F
```

For pixel `(x,y)`, normalized screen coordinates are:

```text
u = 2 * (x + 0.5) / width  - 1
v = 1 - 2 * (y + 0.5) / height
```

With `a = width / height` and `h = tan(FOV / 2)`, the primary direction is:

```text
D = normalize(F + R * (u*a*h) + U * (v*h))
```

The `+0.5` terms sample pixel centers. The exact viewer evaluates this once
for every output pixel. No block reconstruction or interpolation is involved.

## 2. Primitive Intersections

### Sphere

For center `C`, radius `r`, and `oc = O-C`:

```text
b = oc dot D
c = oc dot oc - r²
t = -b - sqrt(discriminant)
```

If the near root is below epsilon, the far root is attempted for rays that
start inside. The normal is:

```text
N = normalize((O + tD) - C)
```

### Axis-aligned box

For each axis, compute the two slab distances and maintain:

```text
t_near = max(axis_near)
```

The ray intersects if `t_near <= t_far` and the selected positive distance is
above epsilon. The axis that supplied `t_near` identifies the face normal.

### Vertical closed cylinder

The side solves the quadratic in the XZ plane:

```text
a = Dx² + Dz²
b = 2 * (Ox*Dx + Oz*Dz)
c = Ox² + Oz² - r²
```

Roots are accepted only when their Y coordinate lies between the base and top.
The two caps use the horizontal plane equation and radial disk test.

### Plane

For horizontal height `y0`:

```text
t = (y0 - Oy) / Dy
```

Parallel rays and distances below epsilon are rejected. Checker variation is
evaluated from `floor(Px) + floor(Pz)`.

## 3. Lighting

The legacy tracer uses direct lighting with bounded recursive reflection:

```text
ambient = 0.14 * albedo
attenuation = power / (distance² + 1)
diffuse = albedo * max(N dot L, 0)
specular = 0.5 * max(N dot H, 0)^shininess
```

where the Blinn half-vector is:

```text
H = normalize(L - D)
```

The final direct term is multiplied component-wise by the light color.
Emissive materials add their emission before direct lighting. Portable red and
blue lights deliberately use pure-color direct fill rather than multiplying
their channels by surface albedo; otherwise red on green becomes yellow and
blue on green becomes cyan.

## 4. Shadows and Reflection

The reference renderer uses five deterministic samples across the light disk:

```text
center, +0.7r*U, -0.7r*U, +0.7r*V, -0.7r*V
```

The fraction of unoccluded samples modulates direct light. The interactive
path uses one shadow query and a bounded interactive mirror environment
approximation to avoid a second full scene traversal per mirror pixel.

Reference recursive reflection is:

```text
R = D - 2 * (D dot N) * N
origin_next = P + epsilon * R
```

The reference path blends reflected radiance using the material reflection
scalar. The material library separately provides Fresnel and transmission
models for the planned BSDF migration.

## 5. PBR/BRDF Research Library

The standalone material library follows the researched microfacet structure:

```text
f_spec = D(h) * G(v,l) * F(v,h) / (4 * NoV * NoL)
```

### Lambert

```text
f_diffuse = base_color / pi
```

### GGX/Trowbridge-Reitz

Perceptual roughness is clamped to `[0.045, 1]` and squared to form alpha.
The normal distribution uses:

```text
D = alpha² / (pi * ((NoH² * (alpha² - 1) + 1)²))
```

### Smith visibility

The implementation uses the separable Smith form with a GGX-compatible
single-direction masking term. This reduces grazing-angle over-brightening.

### Dielectric Fresnel

Normal-incidence reflectance is:

```text
F0 = ((1 - eta) / (1 + eta))²
```

The general-purpose approximation is Schlick:

```text
F = F0 + (1 - F0) * (1 - cos_theta)^5
```

For glass with IOR 1.5, `F0` is approximately 0.04.

### Conductor Fresnel

Conductors use RGB `eta` and `k` optical constants. This produces colored
reflection for gold, copper, brass, and bronze without treating metal color as
diffuse pigment.

### Refraction

Snell's law uses:

```text
eta = eta_incident / eta_transmitted
k = 1 - eta² * (1 - cos_theta²)
```

If `k < 0`, total internal reflection occurs. Otherwise:

```text
T = eta * I + (eta*cos_theta - sqrt(k)) * N
```

## 6. BVH Acceleration

Finite objects are placed in a static median-split bounding volume hierarchy.
Planes remain outside the hierarchy because they are infinite.

Each node stores:

```c
V3 min, max;
int left, right;
int start, count;
```

Construction chooses the largest centroid extent, insertion-sorts the object
indices on that axis, and recursively splits at the midpoint. Leaf size is
four objects. Traversal uses a fixed local integer stack and prunes a node when
its slab interval does not intersect the current nearest distance.

The BVH is rebuilt after moving red/blue balls. This is acceptable because the
interactive scene is compact and movement is infrequent relative to pixel
work. The larger world uses the same acceleration structure after assembly.

## 7. Exact 1x1 AVX2 Renderer

The exact interactive renderer evaluates eight neighboring pixels in an AVX2
packet:

```text
lane 0 -> pixel x
lane 1 -> pixel x+1
...
lane 7 -> pixel x+7
```

Each lane has its own ray direction, nearest distance, normal, object ID, and
RGB output. The packet path vectorizes primary ray generation and box, sphere,
and plane intersection arithmetic. Scalar shading consumes the packet hit
records without running a second primary intersection.

This is not a 3x3 or sparse image technique. Every output pixel is still
computed independently. The optimization is data parallelism inside the CPU.

The build uses:

```text
-O3 -march=native -mtune=native -ffast-math -fno-math-errno -fopenmp
```

These flags are used only for the native interactive target. The reference
headless renderer continues to build with conservative C11 optimization flags.

## 8. Memory and Cache Choices

- Scene objects and BVH nodes are fixed arrays inside `Scene`; this avoids
  per-ray allocation.
- Exact output buffers are heap allocated once by the viewer.
- Packet lanes use stack arrays of eight floats, small enough to stay hot.
- The gamma conversion path uses a 4096-entry lookup table in the AVX2 build.
- Dynamic ball movement rebuilds only scene acceleration data, not framebuffer
  storage.
- The old 3.5 MB per-frame stack allocation bug in `world_viewer` was removed;
  the RGBA upload buffer is persistent heap storage.

## 9. Threading and Benchmarks

OpenMP parallelizes independent image rows. The current machine is an Intel
i5-7300U with two physical cores and four logical processors. Native builds
use four workers because that was the measured best point for the packet path.

Measured milestones on this machine:

| Path | Result |
| --- | ---: |
| Original 240x135 scalar interactive | approximately 4-5 FPS |
| 1280x720 sparse reconstruction | 45-60 FPS, visibly non-exact |
| 1280x720 exact scalar path | approximately 7 FPS |
| 1280x720 exact AVX2 packet path | approximately 45 FPS |

The last number is the current target benchmark. FPS varies with OS scheduling,
background processes, and camera composition. The viewer HUD reports measured
render time for the current frame.

## 10. Numerical and Visual Tradeoffs

- Single precision is used throughout for speed and consistency with the
  existing code.
- `-ffast-math` is restricted to the native interactive target. It may change
  edge-case floating-point behavior and must not be used for determinism tests.
- Interactive mirrors use environment reflection; headless/reference renders
  retain recursive reflection.
- Interactive colored lights are local pure-color emitters by design.
- The compact scene is used for exact 45 FPS; the 70-object world remains the
  quality/reference diorama.

## 11. Integrated Map Textures

The interactive `scene_small()` map now assigns deterministic procedural
textures directly to its legacy tracer materials:

- Ground: world-space checker pattern.
- Dirt foundation: low-frequency hash variation.
- Grass island: brighter green hash variation.
- Plaza stone: gray stone grain.
- Both house walls: subtle mottling that preserves their base colors.
- Roof slabs: repeated dark tile bands with small variation.

Texture evaluation happens after the primary hit and before ambient/direct
lighting. It uses integer cell coordinates and a small integer hash, so it has
no image assets, no runtime allocation, no random state, and identical output
across reference, scalar interactive, and AVX2 packet shading paths. The map
texture descriptor lives in `Material` as a texture ID, scale, strength, and
alternate color.

## 12. Generic Model Mesh Path

The map currently demonstrates the system with the Stanford Bunny
`bun_zipper_res2.ply` mesh, containing 16,301 triangles. The engine API is
generic: `rt_load_mesh()` accepts supported PLY and OBJ files, stores multiple
mesh assets per scene, and creates one mesh object per loaded model. The mesh
is transformed into scene space during loading.

Triangle intersections use a second BVH local to each mesh, with leaves of
eight triangles. This prevents a detailed model from turning every pixel into
a full triangle-list scan.

The interactive renderer keeps the full 1280x720 framebuffer and exact primary
pixel grid. For the mesh scene, it renders the map background through the
native AVX2 exact path, then rasterizes the bunny triangles into the same full
resolution RGB/depth buffer. This is an optimization of scene workload, not a
reduction in display resolution or a block reconstruction technique.
