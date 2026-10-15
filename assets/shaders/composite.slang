#include "common.slang"

struct PushConstant {
    uint composite_id;
    uint light_id;
    uint sdf_id;
    uint scene_id;
    float rim;
}

[[vk::binding(0, 0)]] RWTexture2D<float4> storageImagesF[];
[[vk::binding(0, 0)]] RWTexture2D<uint4> storageImagesU[];
[[vk::binding(4, 0)]] ConstantBuffer<UBO> uniformBuffers[];
[[vk::binding(1, 0)]] StructuredBuffer<Material> storageBuffers[];

float sdf_at(uint id, int2 p, int2 size) {
    return storageImagesF[id].Load(clamp(p, int2(0), size - 1)).x;
}

float3 light_at(uint id, int2 p, int2 size) {
    return storageImagesF[id].Load(clamp(p, int2(0), size - 1)).rgb;
}

[shader("compute")]
[numthreads(8, 8)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID, uniform PushConstant pc) {
    int2 local = int2(dispatchThreadID.xy);
    UBO ubo = uniformBuffers[0];
    int2 size = int2(ubo.scene_width, ubo.scene_height);
    if (any(local >= size)) return;

    uint8_t mat_id = uint8_t(storageImagesU[pc.scene_id].Load(local).x);
    float3 light = storageImagesF[pc.light_id].Load(local).rgb;

    if (mat_id == 0) {
        storageImagesF[pc.composite_id].Store(local, float4(light, 1.0));
        return;
    }
    
    Material mat = storageBuffers[0][mat_id];
    float dist = storageImagesF[pc.sdf_id].Load(local).x;

    float rim = exp(min(dist, 0.0) / pc.rim);
    float3 color = mat.base_color * light * rim + mat.radiance;

    storageImagesF[pc.composite_id].Store(local, float4(color, 1.0));
}