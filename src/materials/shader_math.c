#include "materials.h"
#include <math.h>

#define RTM_PI 3.14159265358979323846f

RtmVec3 rtm_v3(float x, float y, float z) { return (RtmVec3){x, y, z}; }
RtmVec3 rtm_add(RtmVec3 a, RtmVec3 b) { return rtm_v3(a.x+b.x, a.y+b.y, a.z+b.z); }
RtmVec3 rtm_sub(RtmVec3 a, RtmVec3 b) { return rtm_v3(a.x-b.x, a.y-b.y, a.z-b.z); }
RtmVec3 rtm_scale(RtmVec3 a, float s) { return rtm_v3(a.x*s, a.y*s, a.z*s); }
RtmVec3 rtm_mul(RtmVec3 a, RtmVec3 b) { return rtm_v3(a.x*b.x, a.y*b.y, a.z*b.z); }
RtmVec3 rtm_lerp(RtmVec3 a, RtmVec3 b, float t) { return rtm_add(rtm_scale(a, 1.0f-t), rtm_scale(b, t)); }
float rtm_dot(RtmVec3 a, RtmVec3 b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
float rtm_length(RtmVec3 a) { return sqrtf(rtm_dot(a, a)); }
RtmVec3 rtm_normalize(RtmVec3 a) {
    float l = rtm_length(a);
    return l > 1e-8f ? rtm_scale(a, 1.0f/l) : rtm_v3(0, 0, 0);
}
float rtm_clamp(float x, float lo, float hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}
int rtm_finite3(RtmVec3 a) {
    return isfinite(a.x) && isfinite(a.y) && isfinite(a.z);
}

float rtm_fresnel_dielectric(float cos_theta, float ior) {
    float c = rtm_clamp(cos_theta, 0.0f, 1.0f);
    float safe_ior = ior > 1.0f ? ior : 1.0f;
    float f0 = (1.0f - safe_ior) / (1.0f + safe_ior);
    f0 *= f0;
    return f0 + (1.0f - f0) * powf(1.0f - c, 5.0f);
}

static float conductor_channel(float c, float eta, float k) {
    float cos2 = c * c;
    float sin2 = 1.0f - cos2;
    float eta2 = eta * eta;
    float k2 = k * k;
    float t0 = eta2 - k2 - sin2;
    float a2b2 = sqrtf(t0*t0 + 4.0f*eta2*k2);
    float a = sqrtf(fmaxf(0.0f, 0.5f*(a2b2 + t0)));
    float t1 = a2b2 + cos2;
    float t2 = 2.0f*a*c;
    float rs = (t1 - t2) / fmaxf(t1 + t2, 1e-6f);
    float t3 = cos2*a2b2 + sin2*sin2;
    float t4 = t2*sin2;
    float rp = rs * (t3 - t4) / fmaxf(t3 + t4, 1e-6f);
    return rtm_clamp(0.5f*(rs + rp), 0.0f, 1.0f);
}

RtmVec3 rtm_fresnel_conductor(float cos_theta, RtmVec3 eta, RtmVec3 k) {
    float c = rtm_clamp(cos_theta, 0.0f, 1.0f);
    return rtm_v3(conductor_channel(c, eta.x, k.x),
                  conductor_channel(c, eta.y, k.y),
                  conductor_channel(c, eta.z, k.z));
}

float rtm_ggx_distribution(float no_h, float roughness) {
    float r = rtm_clamp(roughness, 0.045f, 1.0f);
    float a = r * r;
    float a2 = a * a;
    float nh = rtm_clamp(no_h, 0.0f, 1.0f);
    float d = nh*nh*(a2 - 1.0f) + 1.0f;
    return a2 / fmaxf(RTM_PI*d*d, 1e-7f);
}

static float smith_g1(float no_v, float roughness) {
    float r = rtm_clamp(roughness, 0.045f, 1.0f);
    float a = r * r;
    float nv = rtm_clamp(no_v, 0.0f, 1.0f);
    float denom = nv + sqrtf(nv*nv + a*a*(1.0f - nv*nv));
    return denom > 1e-7f ? 2.0f*nv/denom : 0.0f;
}

float rtm_smith_visibility(float no_v, float no_l, float roughness) {
    return smith_g1(no_v, roughness) * smith_g1(no_l, roughness);
}
