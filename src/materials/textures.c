#include "materials.h"
#include <math.h>

static float fractf_(float x) { return x - floorf(x); }

float rtm_hash3(RtmVec3 p) {
    float n = sinf(p.x*127.1f + p.y*311.7f + p.z*74.7f) * 43758.5453f;
    return fractf_(n);
}

float rtm_noise3(RtmVec3 p) {
    RtmVec3 i = rtm_v3(floorf(p.x), floorf(p.y), floorf(p.z));
    RtmVec3 f = rtm_v3(fractf_(p.x), fractf_(p.y), fractf_(p.z));
    f = rtm_mul(f, rtm_mul(f, rtm_sub(rtm_v3(3,3,3), rtm_scale(f, 2.0f))));
    float c000 = rtm_hash3(i);
    float c100 = rtm_hash3(rtm_add(i, rtm_v3(1,0,0)));
    float c010 = rtm_hash3(rtm_add(i, rtm_v3(0,1,0)));
    float c110 = rtm_hash3(rtm_add(i, rtm_v3(1,1,0)));
    float c001 = rtm_hash3(rtm_add(i, rtm_v3(0,0,1)));
    float c101 = rtm_hash3(rtm_add(i, rtm_v3(1,0,1)));
    float c011 = rtm_hash3(rtm_add(i, rtm_v3(0,1,1)));
    float c111 = rtm_hash3(rtm_add(i, rtm_v3(1,1,1)));
    float x00 = c000 + (c100-c000)*f.x;
    float x10 = c010 + (c110-c010)*f.x;
    float x01 = c001 + (c101-c001)*f.x;
    float x11 = c011 + (c111-c011)*f.x;
    return (x00 + (x10-x00)*f.y) + ((x01 + (x11-x01)*f.y) -
           (x00 + (x10-x00)*f.y))*f.z;
}

float rtm_fbm(RtmVec3 p, int octaves) {
    float total = 0.0f, amp = 0.5f, norm = 0.0f;
    for (int i = 0; i < octaves; i++) {
        total += rtm_noise3(p) * amp;
        norm += amp;
        p = rtm_scale(p, 2.03f);
        amp *= 0.5f;
    }
    return norm > 0.0f ? total/norm : 0.0f;
}

RtmVec3 rtm_texture_checker(RtmVec3 p, RtmVec3 a, RtmVec3 b, float scale) {
    int c = (int)floorf(p.x*scale) + (int)floorf(p.y*scale) + (int)floorf(p.z*scale);
    return (c & 1) ? b : a;
}

RtmVec3 rtm_texture_stripes(RtmVec3 p, RtmVec3 a, RtmVec3 b, float scale) {
    return fractf_(p.x*scale) < 0.5f ? a : b;
}

RtmVec3 rtm_texture_wood(RtmVec3 p, RtmVec3 light, RtmVec3 dark) {
    float radius = sqrtf(p.x*p.x + p.z*p.z) * 2.2f + rtm_fbm(rtm_scale(p, 3.0f), 3)*0.35f;
    float bands = 0.5f + 0.5f*sinf(radius*18.0f);
    return rtm_lerp(dark, light, rtm_clamp(0.2f + 0.8f*bands, 0, 1));
}

RtmVec3 rtm_texture_marble(RtmVec3 p, RtmVec3 a, RtmVec3 b) {
    float n = p.x*2.0f + rtm_fbm(rtm_scale(p, 2.5f), 4)*2.5f;
    return rtm_lerp(a, b, 0.5f + 0.5f*sinf(n));
}

RtmVec3 rtm_texture_brick(RtmVec3 p, RtmVec3 brick, RtmVec3 mortar) {
    float row = floorf(p.y*3.0f);
    float offset = ((int)row & 1) ? 0.5f : 0.0f;
    float cellx = fractf_(p.x*3.0f + offset);
    float cellz = fractf_(p.z*2.0f);
    return (cellx < 0.08f || cellz < 0.08f) ? mortar : brick;
}

RtmVec3 rtm_texture_rust(RtmVec3 p, RtmVec3 metal, RtmVec3 rust) {
    float n = rtm_fbm(rtm_scale(p, 4.0f), 4);
    return rtm_lerp(metal, rust, rtm_clamp((n-0.45f)*3.0f, 0, 1));
}

RtmVec3 rtm_texture_fabric(RtmVec3 p, RtmVec3 a, RtmVec3 b) {
    float x = fractf_(p.x*28.0f), z = fractf_(p.z*28.0f);
    return ((x < 0.48f) ^ (z < 0.48f)) ? a : b;
}
