#include "materials.h"
#include <math.h>

RtmVec3 rtm_eval_subsurface(const ShaderMaterial *m, RtmVec3 n, RtmVec3 wi, RtmVec3 wo) {
    float no_v = rtm_clamp(rtm_dot(n, wo), 0.0f, 1.0f);
    float wrap = rtm_clamp((rtm_dot(n, wi) + 0.35f) / 1.35f, 0.0f, 1.0f);
    float backlight = (1.0f - no_v) * wrap * m->subsurface;
    RtmVec3 warm = rtm_mul(m->base_color, m->absorption);
    return rtm_add(rtm_lambert(m->base_color), rtm_scale(warm, backlight));
}
