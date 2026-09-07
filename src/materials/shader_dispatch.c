#include "materials.h"

RtmVec3 rtm_eval_bsdf(const ShaderMaterial *m, RtmVec3 n, RtmVec3 wi, RtmVec3 wo) {
    if (!m) return rtm_v3(0, 0, 0);
    switch (m->shader) {
    case RTM_SHADER_LAMBERT:
        return rtm_lambert(m->base_color);
    case RTM_SHADER_CONDUCTOR:
    case RTM_SHADER_DIELECTRIC: {
        RtmVec3 diffuse = rtm_scale(rtm_lambert(m->base_color),
                                    (1.0f - m->metallic) * (1.0f - m->transmission));
        return rtm_add(diffuse, rtm_eval_reflection(m, n, wi, wo));
    }
    case RTM_SHADER_THIN_DIELECTRIC:
        return rtm_eval_reflection(m, n, wi, wo);
    case RTM_SHADER_LAYERED:
        return rtm_add(rtm_scale(rtm_lambert(m->base_color), 1.0f-m->transmission),
                       rtm_eval_layered(m, n, wi, wo));
    case RTM_SHADER_SHEEN:
        return rtm_eval_sheen(m, n, wi, wo);
    case RTM_SHADER_SUBSURFACE:
        return rtm_eval_subsurface(m, n, wi, wo);
    case RTM_SHADER_EMISSIVE:
        return rtm_v3(0, 0, 0);
    }
    return rtm_v3(0, 0, 0);
}

RtmVec3 rtm_emission(const ShaderMaterial *m) {
    return m ? rtm_scale(m->emission, 1.0f) : rtm_v3(0, 0, 0);
}
