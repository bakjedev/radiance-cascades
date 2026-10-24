#include "common.slang"

struct PushConstant {
    uint composite_id;
    uint light_id;
    uint sdf_id;
    uint scene_col_id;
    uint scene_rad_id;
    float rim;
}

[[vk::binding(0, 0)]] RWTexture2D<float4> storageImagesF[];
[[vk::binding(4, 0)]] ConstantBuffer<UBO> uniformBuffers[];

[[vk::binding(2, 0)]] Texture2D textures[];

[shader("compute")]
[numthreads(8, 8)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID, uniform PushConstant pc) {
    int2 local = int2(dispatchThreadID.xy);
    UBO ubo = uniformBuffers[0];
    int2 size = int2(ubo.scene_width, ubo.scene_height);
    if (any(local >= size)) return;

    float4 color = textures[pc.scene_col_id].Load(int3(local, 0));
    float3 light = storageImagesF[pc.light_id].Load(local).rgb;

    if (!(bool)color.a) {
        storageImagesF[pc.composite_id].Store(local, float4(light, 1.0));
        return;
    }
    
    float3 radiance = textures[pc.scene_rad_id].Load(int3(local, 0)).xyz;
    float dist = storageImagesF[pc.sdf_id].Load(local).x;

    float rim = exp(min(dist, 0.0) / pc.rim);
    float3 result = color.xyz * light * rim + radiance;

    storageImagesF[pc.composite_id].Store(local, float4(result, 1.0));
}