# Material Shader Library

This directory is the material/shader layer planned for the CPU ray tracer. It
is intentionally standalone while the API is being stabilized. The current
world renderer still uses its old material type; `material_test.c` exercises
this library without changing the existing render output.

## Design

The library separates four concerns:

1. `ShaderMaterial` stores physical and artistic parameters.
2. `shader_*.c` files implement reusable reflection and transmission models.
3. `textures.c` evaluates procedural values from object-space coordinates.
4. `presets.c` gives named materials that are parameter combinations, not
   one-off shader branches.

The primary opaque response is an energy-aware GGX model:

```text
f = diffuse + D(GGX) * G(Smith) * F(Fresnel) / (4 NoL NoV)
```

Roughness is perceptual. The implementation clamps it to `0.045..1` and uses
its square as the GGX alpha value. Dielectrics use an IOR-derived F0;
conductors use RGB eta/k values, which gives gold and copper colored highlights
without inventing diffuse metallic color.

Transmission is currently represented as a deterministic delta event. The
library exposes exact dielectric Fresnel, Snell refraction, and total internal
reflection checks. This is suitable for the planned deterministic hybrid
tracer. A later path-tracing backend can use the same parameters with sampled
microfacet transmission.

## Files

- `materials.h`: public API and data types.
- `shader_math.c`: vector helpers, Fresnel, GGX distribution, Smith masking.
- `shader_diffuse.c`: Lambert diffuse response.
- `shader_microfacet.c`: dielectric and conductor GGX reflection.
- `shader_transmission.c`: dielectric Fresnel, reflection/refraction events.
- `shader_layered.c`: clearcoat and practical layered response helpers.
- `shader_sheen.c`: cloth/velvet sheen approximation.
- `shader_subsurface.c`: bounded warm subsurface/backlight approximation.
- `shader_dispatch.c`: common material response dispatcher.
- `textures.c`: checker, stripe, noise, wood, marble, brick, rust, and fabric.
- `presets.c`: 40 named material presets.
- `material_test.c`: deterministic unit-style validation harness.

## Shader families

| Family | Intended materials | Current model |
| --- | --- | --- |
| Lambert | matte, chalk, paper, concrete | diffuse only |
| Dielectric GGX | plastic, ceramic, rubber, painted wood | neutral Fresnel specular + diffuse |
| Conductor GGX | gold, copper, aluminum, steel, chrome | RGB eta/k Fresnel |
| Thin dielectric | window glass, acrylic | Fresnel + deterministic transmission |
| Layered | varnish, wet stone, glazed ceramic, car paint | base response + clearcoat |
| Sheen | cotton, canvas, velvet, silk-like surfaces | grazing-angle cloth lobe |
| Subsurface | wax, marble, leaf, snow-like surfaces | bounded backlight approximation |
| Emissive | lamps and glowing objects | emitted radiance |

## Preset catalog

The preset IDs are stable API values and are listed in `presets.c`:

1. Matte White
2. Matte Black
3. Chalk
4. Concrete
5. Terracotta
6. Ceramic
7. Porcelain
8. Rubber
9. Plastic
10. Painted Wood
11. Raw Wood
12. Varnished Wood
13. Paper
14. Cardboard
15. Leather
16. Cotton
17. Canvas
18. Velvet
19. Aluminum
20. Brushed Aluminum
21. Steel
22. Stainless Steel
23. Chrome
24. Copper
25. Brass
26. Gold
27. Bronze
28. Silver
29. Clear Glass
30. Frosted Glass
31. Acrylic
32. Water
33. Ice
34. Diamond
35. Crystal
36. Wet Stone
37. Wet Dirt
38. Wax
39. Marble
40. Leaf
41. Emissive Lamp

The count is 41 because the initial research list included both crystal and
the final emissive lamp entry. Presets deliberately reuse shader families:
gold and copper are not separate implementations from silver; only their
optical constants differ.

## Integration order

1. Convert the tracer's legacy material constructor to `ShaderMaterial`.
2. Add `SurfacePoint` data, including front-face state and object-space point.
3. Replace Blinn-Phong with `rtm_eval_reflection` plus direct-light weighting.
4. Add bounded reflection/transmission events from `shader_transmission.c`.
5. Move checker patterns to `textures.c`, then add texture-driven roughness and
   base color.
6. Add medium tracking before nested glass/water scenes.

No compatibility layer is hidden in this folder. The migration should be an
explicit change in `tracer.h`, `tracer.c`, and `tracer_scene.c` after the
standalone test harness is accepted.
