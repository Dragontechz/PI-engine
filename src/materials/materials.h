#ifndef RT_MATERIALS_H
#define RT_MATERIALS_H

#include <stddef.h>

typedef struct { float x, y, z; } RtmVec3;

typedef enum {
    RTM_SHADER_LAMBERT = 0,
    RTM_SHADER_DIELECTRIC,
    RTM_SHADER_CONDUCTOR,
    RTM_SHADER_THIN_DIELECTRIC,
    RTM_SHADER_EMISSIVE,
    RTM_SHADER_LAYERED,
    RTM_SHADER_SHEEN,
    RTM_SHADER_SUBSURFACE
} RtmShader;

typedef struct {
    RtmShader shader;
    RtmVec3 base_color;
    RtmVec3 emission;
    RtmVec3 absorption;
    RtmVec3 eta;
    RtmVec3 k;
    float roughness;
    float metallic;
    float ior;
    float transmission;
    float clearcoat;
    float clearcoat_roughness;
    float sheen;
    float subsurface;
    float thickness;
} ShaderMaterial;

typedef struct {
    RtmVec3 reflection;
    RtmVec3 transmission;
    RtmVec3 emission;
    float fresnel;
    int reflected;
    int refracted;
    int total_internal_reflection;
} RtmResponse;

typedef enum {
    RTM_PRESET_MATTE_WHITE = 0,
    RTM_PRESET_MATTE_BLACK,
    RTM_PRESET_CHALK,
    RTM_PRESET_CONCRETE,
    RTM_PRESET_TERRACOTTA,
    RTM_PRESET_CERAMIC,
    RTM_PRESET_PORCELAIN,
    RTM_PRESET_RUBBER,
    RTM_PRESET_PLASTIC,
    RTM_PRESET_PAINTED_WOOD,
    RTM_PRESET_RAW_WOOD,
    RTM_PRESET_VARNISHED_WOOD,
    RTM_PRESET_PAPER,
    RTM_PRESET_CARDBOARD,
    RTM_PRESET_LEATHER,
    RTM_PRESET_COTTON,
    RTM_PRESET_CANVAS,
    RTM_PRESET_VELVET,
    RTM_PRESET_ALUMINUM,
    RTM_PRESET_BRUSHED_ALUMINUM,
    RTM_PRESET_STEEL,
    RTM_PRESET_STAINLESS_STEEL,
    RTM_PRESET_CHROME,
    RTM_PRESET_COPPER,
    RTM_PRESET_BRASS,
    RTM_PRESET_GOLD,
    RTM_PRESET_BRONZE,
    RTM_PRESET_SILVER,
    RTM_PRESET_CLEAR_GLASS,
    RTM_PRESET_FROSTED_GLASS,
    RTM_PRESET_ACRYLIC,
    RTM_PRESET_WATER,
    RTM_PRESET_ICE,
    RTM_PRESET_DIAMOND,
    RTM_PRESET_CRYSTAL,
    RTM_PRESET_WET_STONE,
    RTM_PRESET_WET_DIRT,
    RTM_PRESET_WAX,
    RTM_PRESET_MARBLE,
    RTM_PRESET_LEAF,
    RTM_PRESET_EMISSIVE_LAMP,
    RTM_PRESET_COUNT
} RtmPreset;

RtmVec3 rtm_v3(float x, float y, float z);
RtmVec3 rtm_add(RtmVec3 a, RtmVec3 b);
RtmVec3 rtm_sub(RtmVec3 a, RtmVec3 b);
RtmVec3 rtm_scale(RtmVec3 a, float s);
RtmVec3 rtm_mul(RtmVec3 a, RtmVec3 b);
RtmVec3 rtm_lerp(RtmVec3 a, RtmVec3 b, float t);
float rtm_dot(RtmVec3 a, RtmVec3 b);
float rtm_length(RtmVec3 a);
RtmVec3 rtm_normalize(RtmVec3 a);
float rtm_clamp(float x, float lo, float hi);
int rtm_finite3(RtmVec3 a);

float rtm_fresnel_dielectric(float cos_theta, float ior);
RtmVec3 rtm_fresnel_conductor(float cos_theta, RtmVec3 eta, RtmVec3 k);
float rtm_ggx_distribution(float no_h, float roughness);
float rtm_smith_visibility(float no_v, float no_l, float roughness);

RtmVec3 rtm_lambert(RtmVec3 color);
RtmVec3 rtm_eval_reflection(const ShaderMaterial *material,
                            RtmVec3 normal, RtmVec3 wi, RtmVec3 wo);
RtmVec3 rtm_eval_transmission(const ShaderMaterial *material,
                              RtmVec3 normal, RtmVec3 wi, RtmVec3 wo);
RtmVec3 rtm_eval_layered(const ShaderMaterial *material,
                         RtmVec3 normal, RtmVec3 wi, RtmVec3 wo);
RtmVec3 rtm_eval_sheen(const ShaderMaterial *material,
                       RtmVec3 normal, RtmVec3 wi, RtmVec3 wo);
RtmVec3 rtm_eval_subsurface(const ShaderMaterial *material,
                            RtmVec3 normal, RtmVec3 wi, RtmVec3 wo);
RtmVec3 rtm_eval_bsdf(const ShaderMaterial *material,
                      RtmVec3 normal, RtmVec3 wi, RtmVec3 wo);
RtmVec3 rtm_emission(const ShaderMaterial *material);

int rtm_refract(RtmVec3 incident_toward_surface, RtmVec3 normal,
                float ior, RtmVec3 *out_direction);
RtmResponse rtm_surface_response(const ShaderMaterial *material,
                                 RtmVec3 normal, RtmVec3 incident_toward_surface,
                                 int entering);

float rtm_hash3(RtmVec3 p);
float rtm_noise3(RtmVec3 p);
float rtm_fbm(RtmVec3 p, int octaves);
RtmVec3 rtm_texture_checker(RtmVec3 p, RtmVec3 a, RtmVec3 b, float scale);
RtmVec3 rtm_texture_stripes(RtmVec3 p, RtmVec3 a, RtmVec3 b, float scale);
RtmVec3 rtm_texture_wood(RtmVec3 p, RtmVec3 light, RtmVec3 dark);
RtmVec3 rtm_texture_marble(RtmVec3 p, RtmVec3 a, RtmVec3 b);
RtmVec3 rtm_texture_brick(RtmVec3 p, RtmVec3 brick, RtmVec3 mortar);
RtmVec3 rtm_texture_rust(RtmVec3 p, RtmVec3 metal, RtmVec3 rust);
RtmVec3 rtm_texture_fabric(RtmVec3 p, RtmVec3 a, RtmVec3 b);

ShaderMaterial rtm_material_preset(RtmPreset preset);
const char *rtm_preset_name(RtmPreset preset);
size_t rtm_preset_count(void);

#endif
