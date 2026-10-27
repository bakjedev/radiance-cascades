#include "common.slang"

struct PushConstant {
    uint light_id;
    uint light_rw_id;
    uint blur_id;
    uint blur_rw_id;
    bool resolving;
    uint radius;
}

[[vk::binding(0, 0)]] RWTexture2D<float4> storageImagesF[];
[[vk::binding(4, 0)]] ConstantBuffer<UBO> uniformBuffers[];
[[vk::binding(2, 0)]] Texture2D textures[];
[[vk::binding(3, 0)]] SamplerState samplers[];

[shader("compute")]
[numthreads(8, 8)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID, uniform PushConstant pc) {
    int2 local = int2(dispatchThreadID.xy);
    UBO ubo = uniformBuffers[0];
    int2 size = int2(ubo.scene_width, ubo.scene_height);
    if (any(local >= size)) return;
    int radius = max(int(pc.radius), 1);
    float sigma = float(radius) / 3.0;
    float two_sigma_squared = 2.0 * sigma * sigma;

    float2 center = float2(local) + 0.5;
    float2 inv_size = 1.0 / float2(size);
    float2 direction = pc.resolving ? float2(0, 1) : float2(1, 0);

    float4 sum = 0.0;
    float sum_weight = 0.0;
    for (int i = -radius; i <= radius; i++) {
        float weight = exp(-float(i * i) / two_sigma_squared);
        float2 uv = (center + float(i) * direction) * inv_size;
        sum += weight * textures[pc.resolving ? pc.blur_id : pc.light_id].SampleLevel(samplers[0], uv, 0);
        sum_weight += weight;
    }

    if (pc.resolving) {
        float4 current = storageImagesF[pc.light_rw_id].Load(local);
        if (current.a == 0.0)
            storageImagesF[pc.light_rw_id].Store(local, sum / sum_weight);
    } else {
        storageImagesF[pc.blur_rw_id].Store(local, sum / sum_weight);
    }
}