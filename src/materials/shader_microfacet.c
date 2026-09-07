#include "materials.h"
#include <math.h>

static RtmVec3 fresnel_for(const ShaderMaterial *m, float voh) {
    if (m->shader == RTM_SHADER_CONDUCTOR) {
        return rtm_fresnel_conductor(voh, m->eta, m->k);
    }
    float f = rtm_fresnel_dielectric(voh, m->ior);
    return rtm_v3(f, f, f);
}

RtmVec3 rtm_eval_reflection(const ShaderMaterial *m,
                            RtmVec3 n, RtmVec3 wi, RtmVec3 wo) {
    float no_l = rtm_dot(n, wi);
    float no_v = rtm_dot(n, wo);
    if (no_l <= 0.0f || no_v <= 0.0f) return rtm_v3(0, 0, 0);
    RtmVec3 h = rtm_normalize(rtm_add(wi, wo));
    float no_h = fmaxf(rtm_dot(n, h), 0.0f);
    float voh = fmaxf(rtm_dot(wo, h), 0.0f);
    float d = rtm_ggx_distribution(no_h, m->roughness);
    float g = rtm_smith_visibility(no_v, no_l, m->roughness);
    RtmVec3 f = fresnel_for(m, voh);
    return rtm_scale(f, d*g/fmaxf(4.0f*no_l*no_v, 1e-6f));
}
