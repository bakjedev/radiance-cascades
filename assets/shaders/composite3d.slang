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
}

[[vk::binding(0, 0)]] RWTexture2D<float4> storageImagesF[];
[[vk::binding(2, 0)]] Texture2D textures[];
[[vk::binding(4, 0)]] ConstantBuffer<FrameData> uniformBuffers[];

float3 octDecode(float2 f)
{
    float3 n = float3(f, 1.0 - abs(f.x) - abs(f.y));
    float t = saturate(-n.z);
    n.xy += float2(n.x >= 0.0 ? -t : t, n.y >= 0.0 ? -t : t);
    return normalize(n);
}

[shader("compute")]
[numthreads(8, 8)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID, uniform PushConstant pc) {
    int2 local = int2(dispatchThreadID.xy);

    float4 albedo = textures[pc.albedo_id].Load(int3(local, 0));

    float2 normal_encoded = textures[pc.normal_id].Load(int3(local, 0)).xy;
    float3 normal = octDecode(normal_encoded);

    float3 pbr = textures[pc.pbr_id].Load(int3(local, 0)).xyz;

    float3 emissive = textures[pc.emissive_id].Load(int3(local, 0)).xyz;

    float depth = textures[pc.depth_id].Load(int3(local, 0)).x;

    if (depth == 0.0) {
        storageImagesF[pc.composite_id].Store(local, float4(0.02, 0.02, 0.02, 1.0));
        return;
    }

    FrameData frame_data = uniformBuffers[pc.frame_id];
    float2 uv = float2(local) + 0.5;
    float2 ndc_xy = uv * 2.0 - 1.0;
    float4 clip = float4(ndc_xy, depth, 1.0);
    float4 world = mul(frame_data.inv_view_proj, clip);
    world.xyz /= world.w;

    storageImagesF[pc.composite_id].Store(local, albedo);
    // storageImagesF[pc.composite_id].Store(local, float4(normal, 1.0));
    // storageImagesF[pc.composite_id].Store(local, float4(pbr, 1.0));
    // storageImagesF[pc.composite_id].Store(local, float4(emissive, 1.0));
    // storageImagesF[pc.composite_id].Store(local, float4(depth.xxx, 1.0));
    // storageImagesF[pc.composite_id].Store(local, world);
}