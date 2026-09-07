#include "materials.h"
#include <math.h>
#include <stdio.h>

static int check(const char *name, int ok) {
    printf("[%s] %s\n", ok ? "PASS" : "FAIL", name);
    return ok ? 0 : 1;
}

static int nearf_(float a, float b, float e) { return fabsf(a-b) <= e; }

int main(void) {
    int fails = 0;
    RtmVec3 n = rtm_v3(0, 1, 0);
    RtmVec3 wi = rtm_normalize(rtm_v3(.35f, .94f, .10f));
    RtmVec3 wo = rtm_normalize(rtm_v3(-.20f, .97f, .12f));

    float f0 = rtm_fresnel_dielectric(1.0f, 1.5f);
    float fg = rtm_fresnel_dielectric(.1f, 1.5f);
    fails += check("dielectric F0 for glass is about 4%", nearf_(f0, .04f, .005f));
    fails += check("dielectric Fresnel rises at grazing angles", fg > f0 && fg > .55f);

    RtmVec3 gold = rtm_fresnel_conductor(1.0f, rtm_v3(.17f,.35f,1.37f), rtm_v3(3.14f,2.53f,1.93f));
    fails += check("conductor Fresnel remains finite", rtm_finite3(gold));
    fails += check("gold Fresnel is color-dependent", fabsf(gold.x-gold.z) > .05f);

    float sharp = rtm_ggx_distribution(1.0f, .05f);
    float rough = rtm_ggx_distribution(1.0f, .80f);
    fails += check("roughness broadens GGX lobe", sharp > rough);

    RtmVec3 refracted;
    fails += check("air-to-glass refraction succeeds", rtm_refract(rtm_v3(0,-1,0), n, 1.5f, &refracted));
    fails += check("normal refraction points through surface", refracted.y < -.99f);
    fails += check("total internal reflection is detected", !rtm_refract(rtm_normalize(rtm_v3(.98f,.20f,0)), n, 1.5f, &refracted));

    ShaderMaterial glass = rtm_material_preset(RTM_PRESET_CLEAR_GLASS);
    RtmResponse response = rtm_surface_response(&glass, n, rtm_v3(0,-1,0), 1);
    fails += check("glass response has reflected event", response.reflected);
    fails += check("glass response has transmitted event", response.refracted);
    fails += check("glass response preserves finite directions", rtm_finite3(response.reflection) && rtm_finite3(response.transmission));

    RtmVec3 sample = rtm_texture_wood(rtm_v3(1.2f,.3f,2.1f), rtm_v3(.8f,.4f,.1f), rtm_v3(.12f,.03f,.01f));
    fails += check("procedural texture is finite", rtm_finite3(sample));
    fails += check("procedural texture is deterministic", rtm_texture_wood(rtm_v3(1.2f,.3f,2.1f), rtm_v3(.8f,.4f,.1f), rtm_v3(.12f,.03f,.01f)).x == sample.x);

    for (int i = 0; i < RTM_PRESET_COUNT; i++) {
        ShaderMaterial m = rtm_material_preset((RtmPreset)i);
        int valid = rtm_finite3(m.base_color) && rtm_finite3(m.emission) &&
                    isfinite(m.roughness) && isfinite(m.ior) && m.roughness >= 0.0f && m.ior >= 1.0f;
        fails += check(rtm_preset_name((RtmPreset)i), valid);
    }

    RtmVec3 bsdf = rtm_eval_bsdf(&glass, n, wi, wo);
    fails += check("BSDF dispatch returns finite output", rtm_finite3(bsdf));
    printf("\n%s (%d failures)\n", fails == 0 ? "ALL MATERIAL TESTS PASSED" : "MATERIAL TESTS FAILED", fails);
    return fails;
}
