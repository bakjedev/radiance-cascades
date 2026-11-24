#include "common.slang"

struct PushConstant {
    uint composite_id;
    uint frame_id;
    uint albedo_id;
    uint normal_id;
    uint pbr_id;
    uint emissive_id;
    uint depth_id;
}

struct FrameData {
    float4x4 inv_view_proj;
    float3 camera_pos;
    float3 sun_dir;
    float3 sun_color;
    float3 ambient;
    float3 ambient_indirect;
}

struct Fragment {
    float3 normal;
    float3 dir;
    float3 color;
    float metallic;
    float roughness;
    float ao;
}

[[vk::binding(0, 0)]] RWTexture2D<float4> storageImagesF[];
[[vk::binding(2, 0)]] Texture2D textures[];
[[vk::binding(4, 0)]] ConstantBuffer<FrameData> uniformBuffers[];

float3 oct_decode(float2 f)
{
    float3 n = float3(f, 1.0 - abs(f.x) - abs(f.y));
    float t = saturate(-n.z);
    n.xy += float2(n.x >= 0.0 ? -t : t, n.y >= 0.0 ? -t : t);
    return normalize(n);
}

float D_GGX(float NoH, float a) {
    float a2 = a * a;
    float d = (NoH * a2 - NoH) * NoH + 1.0;
    return a2 / (PI * d * d);
}

float V_SmithGGXCorrelated(float NoV, float NoL, float a) {
    float a2 = a * a;
    float gv = NoL * sqrt(NoV * NoV * (1.0 - a2) + a2);
    float gl = NoV * sqrt(NoL * NoL * (1.0 - a2) + a2);
    return 0.5 / (gv + gl);
}

float3 F_Schlick(float VoH, float3 f0) {
    float f = pow(1.0 - VoH, 5.0);
    return f + f0 * (1.0 - f);
}

float Fd_Lambert() { return 1.0 / PI; }

float3 evaluate(Fragment frag, float3 light) {
    float3 H  = normalize(frag.dir + light);
    float NoV = abs(dot(frag.normal, frag.dir)) + 1e-5;
    float NoL = saturate(dot(frag.normal, light));
    float NoH = saturate(dot(frag.normal, H));
    float VoH = saturate(dot(frag.dir, H));
    if (NoL <= 0.0) return 0.0;

    float a = frag.roughness * frag.roughness;
    float3 f0 = lerp(float3(0.04), frag.color, frag.metallic);

    float  D = D_GGX(NoH, a);
    float  V = V_SmithGGXCorrelated(NoV, NoL, a);
    float3 F = F_Schlick(VoH, f0);

    float3 Fr = D * V * F;
    float3 Fd = frag.color * (1.0 - frag.metallic) * Fd_Lambert();

    return (Fd + Fr) * NoL;
}

[shader("compute")]
[numthreads(8, 8)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID, uniform PushConstant pc) {
    int2 local = int2(dispatchThreadID.xy);
    RWTexture2D<float4> img = storageImagesF[pc.composite_id];
    uint2 size;
    img.GetDimensions(size.x, size.y);
    if (any(local >= size)) return;

    int3 pixel = int3(local, 0);

    float depth = textures[pc.depth_id].Load(pixel).x;

    if (depth == 0.0) {
        storageImagesF[pc.composite_id].Store(local, float4(0.02, 0.02, 0.02, 1.0));
        return;
    }

    FrameData frame_data = uniformBuffers[pc.frame_id];
    float2 uv = (float2(local) + 0.5) / float2(size);
    float2 ndc_xy = uv * 2.0 - 1.0;
    float4 clip = float4(ndc_xy, depth, 1.0);
    float4 world = mul(frame_data.inv_view_proj, clip);
    world.xyz /= world.w;

    float4 albedo = textures[pc.albedo_id].Load(pixel);
    float2 normal_encoded = textures[pc.normal_id].Load(pixel).xy;
    float3 pbr = textures[pc.pbr_id].Load(pixel).xyz;
    float3 emissive = textures[pc.emissive_id].Load(pixel).xyz;

    Fragment fragment;
    fragment.normal = oct_decode(normal_encoded);
    fragment.dir = normalize(frame_data.camera_pos - world.xyz);
    fragment.color = albedo.rgb;
    fragment.metallic = pbr.r;
    fragment.roughness = clamp(pbr.g, 0.045, 1.0);
    fragment.ao = pbr.b;

    float3 color = 0.0;

    color += evaluate(fragment, frame_data.sun_dir) * frame_data.sun_color;

    float3 ambient = lerp(frame_data.ambient_indirect, frame_data.ambient, fragment.normal.y * 0.5 + 0.5);
    color += ambient * fragment.color * (1.0 - fragment.metallic) * fragment.ao;

    color += emissive;

    img.Store(local, float4(color, 1.0));
    // img.Store(local, float4(fragment.normal, 1.0));
    // img.Store(local, float4(fragment.color, 1.0));
    // img.Store(local, float4(fragment.metallic, 0.0, 0.0, 1.0));
    // img.Store(local, float4(0.0, fragment.roughness, 0.0, 1.0));
    // img.Store(local, float4(fragment.ao.xxx, 1.0));
    // img.Store(local, float4(emissive, 1.0));
    // img.Store(local, float4(depth.xxx, 1.0));
    // img.Store(local, world);
}