#include "materials.h"
#include <math.h>

RtmVec3 rtm_eval_layered(const ShaderMaterial *m, RtmVec3 n, RtmVec3 wi, RtmVec3 wo) {
    RtmVec3 base = rtm_eval_reflection(m, n, wi, wo);
    float no_l = rtm_clamp(rtm_dot(n, wi), 0.0f, 1.0f);
    float no_v = rtm_clamp(rtm_dot(n, wo), 0.0f, 1.0f);
    float grazing = powf(1.0f - fminf(no_l, no_v), 5.0f);
    ShaderMaterial coat = *m;
    coat.shader = RTM_SHADER_DIELECTRIC;
    coat.base_color = rtm_v3(1, 1, 1);
    coat.ior = 1.5f;
    coat.roughness = m->clearcoat_roughness;
    RtmVec3 clear = rtm_eval_reflection(&coat, n, wi, wo);
    return rtm_add(base, rtm_scale(clear, m->clearcoat * (0.04f + 0.96f*grazing)));
}
