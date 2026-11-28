#include "common.slang"

struct VSInput {
    float3 pos : POSITION;
    float3 normal : NORMAL;
    float2 uv : TEXCOORD0;
    float2 uv1 : TEXCOORD1;
    float4 tangent: TANGENT;
}

struct VSOutput {
    float4 pos : SV_Position;
    float3 normal : NORMAL;
    float2 uv : TEXCOORD0;
    float2 uv1 : TEXCOORD1;
    float4 tangent: TANGENT;
    nointerpolation uint material;
}

struct FSOutput {
    float4 color : SV_Target0;
    float2 normal : SV_Target1;
    float4 pbr : SV_Target2;
    float3 emissive : SV_Target3;
}

struct PushConstant {
    float4x4 view_proj;
    uint geometries_id;
    uint materials_id;
    uint textures_id;
    uint samplers_id;
}

[[vk::binding(1, 0)]] StructuredBuffer<Geometry> storageBuffersGeometry[];

// https://www.shadertoy.com/view/3s33zj
float3x3 adjugate(float3x3 m) {
    return float3x3(
        cross(m[1], m[2]),
        cross(m[2], m[0]),
        cross(m[0], m[1])
    );
}

[shader("vertex")]
VSOutput vs_main(VSInput input, uint instanceID : SV_StartInstanceLocation, uniform PushConstant pc) {
    VSOutput out;
    Geometry geometry = storageBuffersGeometry[pc.geometries_id][instanceID];

    float4 world = mul(geometry.transform,float4(input.pos, 1.0));
    out.pos = mul(  pc.view_proj, world);

    float3x3 normal = adjugate((float3x3)geometry.transform);
    out.normal = normalize(mul(normal, input.normal));

    out.tangent = float4(normalize(mul((float3x3)geometry.transform, input.tangent.xyz)), input.tangent.w);

    out.uv = input.uv;
    out.uv1 = input.uv1;

    out.material = geometry.material;

    return out;
}

struct Material {
    float4 base_color;
    float3 emissive_color;
    float metallic;
    float roughness;
    float normal_scale;
    float occlusion_scale;
    uint albedo_tex;
    uint metal_roughness_tex;
    uint normal_tex;
    uint occlusion_tex;
    uint emissive_tex;
    bool alpha_mode;
    float alpha_cutoff;
}

[[vk::binding(1, 0)]] StructuredBuffer<Material> storageBuffersMaterial[];

[[vk::binding(2, 0)]] Texture2D textures[];
[[vk::binding(3, 0)]] SamplerState samplers[];

static const uint MaterialTextureBits = 23u;
static const uint MaterialTextureMask = (1u << MaterialTextureBits) - 1u;
static const uint MaterialSamplerShift  = MaterialTextureBits + 1u;

struct Texture {
    uint index;
    uint uv;
    uint sampler;
};

bool try_get_texture(uint packed, out Texture texture) {
  texture.index = packed & MaterialTextureMask;
  texture.uv = (packed >> MaterialTextureBits) & 1u;
  texture.sampler = packed >> MaterialSamplerShift;
  return texture.index != MaterialTextureMask;
}

float2 sign_not_zero(float2 v) {
    return float2(v.x >= 0.0 ? 1.0 : -1.0, v.y >= 0.0 ? 1.0 : -1.0);
}

float2 oct_encode(float3 n) {
    n /= (abs(n.x) + abs(n.y) + abs(n.z));
    float2 p = n.xy;
    if (n.z < 0.0)
        p = (1.0 - abs(p.yx)) * sign_not_zero(p);
    return p;
}

[shader("fragment")]
FSOutput fs_main(VSOutput input, uniform PushConstant pc) {
    FSOutput out;
    Material material = storageBuffersMaterial[pc.materials_id][input.material];

    float4 albedo = material.base_color;
    Texture albedo_tex;
    if (try_get_texture(material.albedo_tex, albedo_tex))
    {
        float2 uv = albedo_tex.uv == 0 ? input.uv : input.uv1;
        albedo *= textures[NonUniformResourceIndex(pc.textures_id + albedo_tex.index)].Sample(
            samplers[NonUniformResourceIndex(pc.samplers_id + albedo_tex.sampler)],
            uv);
    }

    out.color = albedo;

    float3 N = normalize(input.normal);
    float3 T = normalize(input.tangent.xyz - N * dot(N, input.tangent.xyz));
    float3 B = cross(N, T) * input.tangent.w;

    float3 normal = float3(0.5, 0.5, 1.0);
    Texture normal_tex;
    if (try_get_texture(material.normal_tex, normal_tex))
    {
        float2 uv = normal_tex.uv == 0 ? input.uv : input.uv1;
        normal = textures[NonUniformResourceIndex(pc.textures_id + normal_tex.index)].Sample(
            samplers[NonUniformResourceIndex(pc.samplers_id + normal_tex.sampler)],
            uv).xyz;
    }
    normal = normal * 2.0 - 1.0;
    normal.xy *= material.normal_scale;
    float3 normal_ws = normalize(normal.x * T + normal.y * B + normal.z * N);

    out.normal = oct_encode(normal_ws);

    float2 metal_roughness = 1.0;
    Texture metal_rough_tex;
    if (try_get_texture(material.metal_roughness_tex, metal_rough_tex))
    {
        float2 uv = metal_rough_tex.uv == 0 ? input.uv : input.uv1;
        metal_roughness = textures[NonUniformResourceIndex(pc.textures_id + metal_rough_tex.index)].Sample(
            samplers[NonUniformResourceIndex(pc.samplers_id + metal_rough_tex.sampler)],
            uv).bg;
    }
    metal_roughness.x *= material.metallic;
    metal_roughness.y *= material.roughness;

    float occlusion = 1.0;
    Texture occlusion_tex;
    if (try_get_texture(material.occlusion_tex, occlusion_tex))
    {
        float2 uv = occlusion_tex.uv == 0 ? input.uv : input.uv1;
        float value = textures[NonUniformResourceIndex(pc.textures_id + occlusion_tex.index)].Sample(
            samplers[NonUniformResourceIndex(pc.samplers_id + occlusion_tex.sampler)],
            uv).r;
        occlusion = 1.0 + material.occlusion_scale * (value - 1.0);
    }

    out.pbr = float4(metal_roughness, occlusion, 1.0);

    float3 emissive = 1.0;
    Texture emissive_tex;
    if (try_get_texture(material.emissive_tex, emissive_tex))
    {
        float2 uv = emissive_tex.uv == 0 ? input.uv : input.uv1;
        emissive = textures[NonUniformResourceIndex(pc.textures_id + emissive_tex.index)].Sample(
            samplers[NonUniformResourceIndex(pc.samplers_id + emissive_tex.sampler)],
            uv).rgb;
    }
    emissive *= material.emissive_color;

    out.emissive = emissive;

    if (material.alpha_mode)
        clip(albedo.a - material.alpha_cutoff);

    return out;
}