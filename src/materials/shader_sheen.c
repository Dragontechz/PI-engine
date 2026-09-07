#include "materials.h"
#include <math.h>

RtmVec3 rtm_eval_sheen(const ShaderMaterial *m, RtmVec3 n, RtmVec3 wi, RtmVec3 wo) {
    float no_l = rtm_clamp(rtm_dot(n, wi), 0.0f, 1.0f);
    float no_v = rtm_clamp(rtm_dot(n, wo), 0.0f, 1.0f);
    float grazing = powf(1.0f - fminf(no_l, no_v), 2.0f);
    RtmVec3 cloth = rtm_scale(m->base_color, m->sheen * grazing / 3.14159265359f);
    return rtm_add(rtm_lambert(m->base_color), cloth);
}
