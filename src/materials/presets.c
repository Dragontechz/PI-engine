#include "materials.h"

static ShaderMaterial base(RtmShader shader, RtmVec3 color, float roughness) {
    ShaderMaterial m = {0};
    m.shader = shader;
    m.base_color = color;
    m.emission = rtm_v3(0, 0, 0);
    m.absorption = rtm_v3(1, 1, 1);
    m.eta = rtm_v3(1, 1, 1);
    m.k = rtm_v3(0, 0, 0);
    m.roughness = roughness;
    m.ior = 1.5f;
    m.transmission = 0.0f;
    m.metallic = shader == RTM_SHADER_CONDUCTOR ? 1.0f : 0.0f;
    m.thickness = 0.0f;
    return m;
}

static ShaderMaterial dielectric(RtmVec3 color, float roughness, float ior) {
    ShaderMaterial m = base(RTM_SHADER_DIELECTRIC, color, roughness);
    m.ior = ior;
    return m;
}

static ShaderMaterial conductor(RtmVec3 color, float roughness,
                                RtmVec3 eta, RtmVec3 k) {
    ShaderMaterial m = base(RTM_SHADER_CONDUCTOR, color, roughness);
    m.eta = eta;
    m.k = k;
    return m;
}

static ShaderMaterial glass(RtmShader shader, RtmVec3 tint, float roughness, float ior) {
    ShaderMaterial m = base(shader, tint, roughness);
    m.ior = ior;
    m.transmission = 1.0f;
    m.thickness = 0.02f;
    return m;
}

ShaderMaterial rtm_material_preset(RtmPreset p) {
    ShaderMaterial m;
    switch (p) {
    case RTM_PRESET_MATTE_WHITE: return base(RTM_SHADER_LAMBERT, rtm_v3(.86f,.86f,.86f), 1.0f);
    case RTM_PRESET_MATTE_BLACK: return base(RTM_SHADER_LAMBERT, rtm_v3(.025f,.025f,.025f), 1.0f);
    case RTM_PRESET_CHALK: return dielectric(rtm_v3(.78f,.77f,.72f), .92f, 1.46f);
    case RTM_PRESET_CONCRETE: return dielectric(rtm_v3(.48f,.49f,.47f), .88f, 1.50f);
    case RTM_PRESET_TERRACOTTA: return dielectric(rtm_v3(.64f,.24f,.12f), .76f, 1.53f);
    case RTM_PRESET_CERAMIC: return dielectric(rtm_v3(.72f,.74f,.77f), .28f, 1.54f);
    case RTM_PRESET_PORCELAIN: return dielectric(rtm_v3(.92f,.91f,.86f), .12f, 1.55f);
    case RTM_PRESET_RUBBER: return dielectric(rtm_v3(.035f,.04f,.038f), .82f, 1.52f);
    case RTM_PRESET_PLASTIC: return dielectric(rtm_v3(.20f,.42f,.78f), .32f, 1.48f);
    case RTM_PRESET_PAINTED_WOOD: return dielectric(rtm_v3(.48f,.16f,.07f), .46f, 1.50f);
    case RTM_PRESET_RAW_WOOD: return dielectric(rtm_v3(.48f,.22f,.08f), .72f, 1.48f);
    case RTM_PRESET_VARNISHED_WOOD:
        m = dielectric(rtm_v3(.36f,.12f,.035f), .48f, 1.48f);
        m.shader = RTM_SHADER_LAYERED; m.clearcoat = .8f; m.clearcoat_roughness = .10f; return m;
    case RTM_PRESET_PAPER: return base(RTM_SHADER_LAMBERT, rtm_v3(.82f,.80f,.72f), .98f);
    case RTM_PRESET_CARDBOARD: return dielectric(rtm_v3(.48f,.28f,.11f), .90f, 1.47f);
    case RTM_PRESET_LEATHER: return dielectric(rtm_v3(.16f,.045f,.018f), .68f, 1.48f);
    case RTM_PRESET_COTTON:
        m = base(RTM_SHADER_SHEEN, rtm_v3(.55f,.58f,.62f), .90f); m.sheen = .36f; return m;
    case RTM_PRESET_CANVAS:
        m = base(RTM_SHADER_SHEEN, rtm_v3(.42f,.36f,.25f), .94f); m.sheen = .23f; return m;
    case RTM_PRESET_VELVET:
        m = base(RTM_SHADER_SHEEN, rtm_v3(.16f,.025f,.09f), .72f); m.sheen = .85f; return m;
    case RTM_PRESET_ALUMINUM: return conductor(rtm_v3(.91f,.93f,.95f), .24f, rtm_v3(1.35f,.96f,.62f), rtm_v3(7.47f,6.40f,5.30f));
    case RTM_PRESET_BRUSHED_ALUMINUM: return conductor(rtm_v3(.91f,.93f,.95f), .56f, rtm_v3(1.35f,.96f,.62f), rtm_v3(7.47f,6.40f,5.30f));
    case RTM_PRESET_STEEL: return conductor(rtm_v3(.72f,.74f,.77f), .38f, rtm_v3(2.90f,2.95f,2.85f), rtm_v3(3.10f,3.20f,3.30f));
    case RTM_PRESET_STAINLESS_STEEL: return conductor(rtm_v3(.80f,.82f,.84f), .30f, rtm_v3(2.80f,2.90f,3.00f), rtm_v3(3.20f,3.30f,3.45f));
    case RTM_PRESET_CHROME: return conductor(rtm_v3(.95f,.95f,.95f), .06f, rtm_v3(3.10f,3.10f,3.10f), rtm_v3(4.10f,4.10f,4.10f));
    case RTM_PRESET_COPPER: return conductor(rtm_v3(.72f,.28f,.12f), .25f, rtm_v3(.27f,.68f,1.32f), rtm_v3(3.61f,2.62f,2.29f));
    case RTM_PRESET_BRASS: return conductor(rtm_v3(.76f,.47f,.14f), .28f, rtm_v3(.44f,.64f,1.04f), rtm_v3(2.54f,2.32f,2.22f));
    case RTM_PRESET_GOLD: return conductor(rtm_v3(.83f,.59f,.18f), .18f, rtm_v3(.17f,.35f,1.37f), rtm_v3(3.14f,2.53f,1.93f));
    case RTM_PRESET_BRONZE: return conductor(rtm_v3(.54f,.28f,.10f), .42f, rtm_v3(.46f,.63f,1.11f), rtm_v3(3.10f,2.70f,2.36f));
    case RTM_PRESET_SILVER: return conductor(rtm_v3(.96f,.96f,.96f), .10f, rtm_v3(.16f,.16f,.16f), rtm_v3(4.10f,3.10f,2.60f));
    case RTM_PRESET_CLEAR_GLASS: return glass(RTM_SHADER_THIN_DIELECTRIC, rtm_v3(.98f,.99f,1.0f), .045f, 1.50f);
    case RTM_PRESET_FROSTED_GLASS: return glass(RTM_SHADER_THIN_DIELECTRIC, rtm_v3(.90f,.95f,1.0f), .55f, 1.50f);
    case RTM_PRESET_ACRYLIC: return glass(RTM_SHADER_THIN_DIELECTRIC, rtm_v3(.86f,.96f,1.0f), .16f, 1.49f);
    case RTM_PRESET_WATER:
        m = glass(RTM_SHADER_DIELECTRIC, rtm_v3(.08f,.42f,.54f), .05f, 1.333f); m.absorption = rtm_v3(.08f,.34f,.52f); m.thickness = .12f; return m;
    case RTM_PRESET_ICE: return glass(RTM_SHADER_DIELECTRIC, rtm_v3(.70f,.88f,1.0f), .28f, 1.31f);
    case RTM_PRESET_DIAMOND: return glass(RTM_SHADER_DIELECTRIC, rtm_v3(1,1,1), .04f, 2.42f);
    case RTM_PRESET_CRYSTAL: return glass(RTM_SHADER_DIELECTRIC, rtm_v3(.82f,.94f,1.0f), .12f, 1.55f);
    case RTM_PRESET_WET_STONE:
        m = dielectric(rtm_v3(.20f,.22f,.23f), .84f, 1.50f); m.shader = RTM_SHADER_LAYERED; m.clearcoat=.65f; m.clearcoat_roughness=.08f; return m;
    case RTM_PRESET_WET_DIRT:
        m = dielectric(rtm_v3(.16f,.07f,.025f), .92f, 1.48f); m.shader = RTM_SHADER_LAYERED; m.clearcoat=.40f; m.clearcoat_roughness=.18f; return m;
    case RTM_PRESET_WAX:
        m = base(RTM_SHADER_SUBSURFACE, rtm_v3(.82f,.22f,.08f), .68f); m.absorption=rtm_v3(1.0f,.52f,.22f); m.subsurface=.65f; return m;
    case RTM_PRESET_MARBLE:
        m = base(RTM_SHADER_SUBSURFACE, rtm_v3(.74f,.76f,.75f), .42f); m.absorption=rtm_v3(.86f,.90f,.96f); m.subsurface=.32f; return m;
    case RTM_PRESET_LEAF:
        m = base(RTM_SHADER_SUBSURFACE, rtm_v3(.08f,.36f,.035f), .62f); m.absorption=rtm_v3(.12f,.60f,.08f); m.subsurface=.85f; m.thickness=.2f; return m;
    case RTM_PRESET_EMISSIVE_LAMP:
        m = base(RTM_SHADER_EMISSIVE, rtm_v3(1.0f,.55f,.16f), 1.0f); m.emission=rtm_v3(8.0f,2.5f,.45f); return m;
    case RTM_PRESET_COUNT: break;
    }
    return base(RTM_SHADER_LAMBERT, rtm_v3(.5f,.5f,.5f), 1.0f);
}

static const char *names[] = {
    "Matte White", "Matte Black", "Chalk", "Concrete", "Terracotta",
    "Ceramic", "Porcelain", "Rubber", "Plastic", "Painted Wood",
    "Raw Wood", "Varnished Wood", "Paper", "Cardboard", "Leather",
    "Cotton", "Canvas", "Velvet", "Aluminum", "Brushed Aluminum",
    "Steel", "Stainless Steel", "Chrome", "Copper", "Brass", "Gold",
    "Bronze", "Silver", "Clear Glass", "Frosted Glass", "Acrylic", "Water",
    "Ice", "Diamond", "Crystal", "Wet Stone", "Wet Dirt", "Wax", "Marble",
    "Leaf", "Emissive Lamp"
};

const char *rtm_preset_name(RtmPreset p) {
    return p >= 0 && p < RTM_PRESET_COUNT ? names[p] : "Unknown";
}

size_t rtm_preset_count(void) { return RTM_PRESET_COUNT; }
