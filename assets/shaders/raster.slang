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
    row_major float4x4 view_proj;
    uint geometries_id;
    uint materials_id;
    uint textures_id;
    uint samplers_id;
}

struct Geometry {
    float4x4 transform;
    uint material;
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

    float4 world = mul(float4(input.pos, 1.0), geometry.transform);
    out.pos = mul(world, pc.view_proj);

    float3x3 normal = adjugate((float3x3)geometry.transform);
    out.normal = normalize(mul(input.normal, normal));

    out.tangent = float4(normalize(mul(input.tangent.xyz, (float3x3)geometry.transform)), input.tangent.w);

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
}

[[vk::binding(1, 0)]] StructuredBuffer<Material> storageBuffersMaterial[];

[[vk::binding(2, 0)]] Texture2D textures[];
[[vk::binding(3, 0)]] SamplerState samplers[];

static const uint MaterialTextureBits = 24u;
static const uint MaterialTextureMask = (1u << MaterialTextureBits) - 1u;

void unpack_texture_id(uint packed, out uint index, out uint tex_coord) {
    index = packed & MaterialTextureMask;
    tex_coord = packed >> MaterialTextureBits;
}

bool try_get_texture(uint texture, out uint index, out uint tex_coord) {
    if (texture == 0xFFFFFFFFu) {
        index = 0u;
        tex_coord = 0u;
        return false;
    }

    unpack_texture_id(texture, index, tex_coord);
    return true;
}

float2 signNotZero(float2 v) {
    return float2(v.x >= 0.0 ? 1.0 : -1.0, v.y >= 0.0 ? 1.0 : -1.0);
}

float2 octEncode(float3 n) {
    n /= (abs(n.x) + abs(n.y) + abs(n.z));
    float2 p = n.xy;
    if (n.z < 0.0)
        p = (1.0 - abs(p.yx)) * signNotZero(p);
    return p;
}

[shader("fragment")]
FSOutput fs_main(VSOutput input, uniform PushConstant pc) {
    FSOutput out;
    Material material = storageBuffersMaterial[pc.materials_id][input.material];

    float4 albedo = material.base_color;
    uint albedo_idx, albedo_uv;
    if (try_get_texture(material.albedo_tex, albedo_idx, albedo_uv))
    {
        float2 uv = albedo_uv == 0 ? input.uv : input.uv1;
        albedo *= textures[NonUniformResourceIndex(albedo_idx)].Sample(samplers[0], uv);
    }

    out.color = albedo;

    float3 N = input.normal;
    float3 T = normalize(input.tangent.xyz - N * dot(N, input.tangent.xyz));
    float3 B = cross(N, T) * input.tangent.w;

    float3 normal;
    uint normal_idx, normal_uv;
    if (try_get_texture(material.normal_tex, normal_idx, normal_uv))
    {
        float2 uv = normal_uv == 0 ? input.uv : input.uv1;
        normal = textures[NonUniformResourceIndex(normal_idx)].Sample(samplers[0], uv).xyz;
    }
    normal *= 2.0 - 1.0;
    normal.xy *= material.normal_scale;
    float3 normal_ws = normalize(normal.x * T + normal.y * B + normal.z * N);

    out.normal = octEncode(normal_ws);

    float2 metal_roughness;
    uint metal_rough_idx, metal_rough_uv;
    if (try_get_texture(material.metal_roughness_tex, metal_rough_idx, metal_rough_uv))
    {
        float2 uv = metal_rough_uv == 0 ? input.uv : input.uv1;
        metal_roughness = textures[NonUniformResourceIndex(metal_rough_idx)].Sample(samplers[0], uv).bg;
    }

    float occlusion;
    uint occlusion_idx, occlusion_uv;
    if (try_get_texture(material.occlusion_tex, occlusion_idx, occlusion_uv))
    {
        float2 uv = occlusion_uv == 0 ? input.uv : input.uv1;
        occlusion = textures[NonUniformResourceIndex(occlusion_idx)].Sample(samplers[0], uv).r;
    }

    out.pbr = float4(metal_roughness, occlusion, 1.0);

    float3 emissive;
    uint emissive_idx, emissive_uv;
    if (try_get_texture(material.emissive_tex, emissive_idx, emissive_uv))
    {
        float2 uv = emissive_uv == 0 ? input.uv : input.uv1;
        emissive = textures[NonUniformResourceIndex(emissive_idx)].Sample(samplers[0], uv).rgb;
    }
    emissive *= material.emissive_color;

    out.emissive = emissive;

    return out;
}