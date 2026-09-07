#include "materials.h"
#include <math.h>

int rtm_refract(RtmVec3 incident, RtmVec3 normal, float ior, RtmVec3 *out) {
    if (!out) return 0;
    RtmVec3 n = rtm_normalize(normal);
    RtmVec3 v = rtm_normalize(incident);
    float cosi = rtm_clamp(rtm_dot(v, n), -1.0f, 1.0f);
    float etai = 1.0f, etat = ior > 1.0f ? ior : 1.0f;
    if (cosi > 0.0f) {
        RtmVec3 tmp = n;
        n = rtm_scale(tmp, -1.0f);
        etai = etat;
        etat = 1.0f;
    } else {
        cosi = -cosi;
    }
    float eta = etai / etat;
    float k = 1.0f - eta*eta*(1.0f - cosi*cosi);
    if (k < 0.0f) return 0;
    *out = rtm_normalize(rtm_add(rtm_scale(v, eta), rtm_scale(n, eta*cosi - sqrtf(k))));
    return 1;
}

RtmVec3 rtm_eval_transmission(const ShaderMaterial *m,
                              RtmVec3 n, RtmVec3 wi, RtmVec3 wo) {
    float no_l = rtm_dot(n, wi);
    float no_v = rtm_dot(n, wo);
    if (no_l * no_v >= 0.0f || m->transmission <= 0.0f) return rtm_v3(0, 0, 0);
    float f = rtm_fresnel_dielectric(fabsf(no_v), m->ior);
    float tint = expf(-fmaxf(m->thickness, 0.0f));
    return rtm_scale(m->base_color, (1.0f - f) * m->transmission * tint);
}

RtmResponse rtm_surface_response(const ShaderMaterial *m, RtmVec3 n,
                                 RtmVec3 incident, int entering) {
    RtmResponse r = { {0,0,0}, {0,0,0}, rtm_emission(m), 0.0f, 0, 0, 0 };
    float c = fabsf(rtm_dot(rtm_normalize(incident), rtm_normalize(n)));
    r.fresnel = rtm_fresnel_dielectric(c, m->ior);
    r.reflected = 1;
    RtmVec3 refl = rtm_sub(incident, rtm_scale(n, 2.0f*rtm_dot(incident, n)));
    r.reflection = rtm_normalize(refl);
    if (m->shader == RTM_SHADER_CONDUCTOR) {
        r.fresnel = (rtm_fresnel_conductor(c, m->eta, m->k).x +
                     rtm_fresnel_conductor(c, m->eta, m->k).y +
                     rtm_fresnel_conductor(c, m->eta, m->k).z) / 3.0f;
    }
    if (m->transmission > 0.0f && m->shader != RTM_SHADER_CONDUCTOR) {
        RtmVec3 transmitted;
        if (rtm_refract(incident, n, m->ior, &transmitted)) {
            r.transmission = transmitted;
            r.refracted = 1;
        } else {
            r.total_internal_reflection = 1;
            r.fresnel = 1.0f;
        }
    }
    (void)entering;
    return r;
}
