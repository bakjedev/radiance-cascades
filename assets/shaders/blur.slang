#include "common.slang"

struct PushConstant {
    uint light_id;
    uint light_rw_id;
    uint blur_id;
    uint blur_rw_id;
    bool resolving;
    uint sample_count;
    uint weights_id;
    uint offsets_id;
}

[[vk::binding(0, 0)]] RWTexture2D<float4> storageImagesF[];
[[vk::binding(1, 0)]] StructuredBuffer<float> storageBuffers[];
[[vk::binding(2, 0)]] Texture2D textures[];
[[vk::binding(3, 0)]] SamplerState samplers[];
[[vk::binding(4, 0)]] ConstantBuffer<UBO> uniformBuffers[];

[shader("compute")]
[numthreads(8, 8)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID, uniform PushConstant pc) {
    int2 local = int2(dispatchThreadID.xy);
    UBO ubo = uniformBuffers[0];
    int2 size = int2(ubo.scene_width, ubo.scene_height);
    if (any(local >= size)) return;

    uint read_idx = pc.resolving ? pc.blur_id : pc.light_id;

    float2 center = float2(local) + 0.5;
    float2 inv_size = 1.0 / float2(size);
    float2 direction = pc.resolving ? float2(0, 1) : float2(1, 0);

    float4 result = textures[read_idx].SampleLevel(samplers[0], center * inv_size, 0) * storageBuffers[pc.weights_id][0];

    for (int i = 1; i < pc.sample_count; i++) {
        float2 uv_1 = (center + direction * storageBuffers[pc.offsets_id][i]) * inv_size;
        result += textures[read_idx].SampleLevel(samplers[0], uv_1, 0) * storageBuffers[pc.weights_id][i];

        float2 uv_2 = (center - direction * storageBuffers[pc.offsets_id][i]) * inv_size;
        result += textures[read_idx].SampleLevel(samplers[0], uv_2, 0) * storageBuffers[pc.weights_id][i];
    }

    if (pc.resolving) {
        float4 current = storageImagesF[pc.light_rw_id].Load(local);
        if (current.a == 0.0)
            storageImagesF[pc.light_rw_id].Store(local, result);
    } else {
        storageImagesF[pc.blur_rw_id].Store(local, result);
    }
}