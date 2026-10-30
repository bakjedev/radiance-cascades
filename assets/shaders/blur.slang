#include "common.slang"

struct BlurPushConstant {
    uint light_id;
    uint blur_id;
    uint blur_rw_id;
    uint sampler_id;
    uint pass;
    uint sample_count;
    uint weights_id;
    uint offsets_id;
}

struct ResolvePushConstant {
    uint light_rw_id;
    uint blur_id;
    uint sampler_id;
}

[[vk::binding(0, 0)]] RWTexture2D<float4> storageImagesF[];
[[vk::binding(0, 0)]] RWTexture2DArray<float4> storageImageArraysF[];
[[vk::binding(1, 0)]] StructuredBuffer<float> storageBuffers[];
[[vk::binding(2, 0)]] Texture2D textures[];
[[vk::binding(3, 0)]] SamplerState samplers[];
[[vk::binding(4, 0)]] ConstantBuffer<UBO> uniformBuffers[];

[shader("compute")]
[numthreads(8, 8)]
void blur(uint3 dispatchThreadID : SV_DispatchThreadID, uniform BlurPushConstant pc) {
    int2 local = int2(dispatchThreadID.xy);
    UBO ubo = uniformBuffers[0];
    int2 size = int2(ubo.scene_width >> 1, ubo.scene_height >> 1);
    if (any(local >= size)) return;

    bool second_pass = bool(pc.pass);

    uint read_idx = second_pass ? pc.blur_id : pc.light_id;

    float2 center = float2(local) + 0.5;
    float2 inv_size = 1.0 / float2(size);
    float2 direction = second_pass ? float2(0, 1) : float2(1, 0);

    float4 result = textures[read_idx].SampleLevel(samplers[pc.sampler_id], center * inv_size, 0) * storageBuffers[pc.weights_id][0];

    for (int i = 1; i < pc.sample_count; i++) {
        float2 uv_1 = (center + direction * storageBuffers[pc.offsets_id][i]) * inv_size;
        result += textures[read_idx].SampleLevel(samplers[pc.sampler_id], uv_1, 0) * storageBuffers[pc.weights_id][i];

        float2 uv_2 = (center - direction * storageBuffers[pc.offsets_id][i]) * inv_size;
        result += textures[read_idx].SampleLevel(samplers[pc.sampler_id], uv_2, 0) * storageBuffers[pc.weights_id][i];
    }

    storageImageArraysF[pc.blur_rw_id].Store(int3(local, pc.pass), result);
}

[shader("compute")]
[numthreads(8, 8)]
void resolve(uint3 dispatchThreadID : SV_DispatchThreadID, uniform ResolvePushConstant pc) {
    int2 local = int2(dispatchThreadID.xy);
    UBO ubo = uniformBuffers[0];
    int2 size = int2(ubo.scene_width, ubo.scene_height);
    if (any(local >= size)) return;

    float2 center = float2(local) + 0.5;
    float2 inv_size = 1.0 / float2(size);

    float4 color = textures[pc.blur_id].SampleLevel(samplers[pc.sampler_id], center * inv_size, 0);

    float4 current = storageImagesF[pc.light_rw_id].Load(local);
    if (current.a == 0.0) {
        storageImagesF[pc.light_rw_id].Store(local, color);
    }
}
