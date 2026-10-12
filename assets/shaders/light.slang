#include "common.slang"

struct PushConstant {
    uint cascades_id;
    uint light_id;
    uint sdf_id;
    float spacing;
    uint probe_size;
}

[[vk::binding(0, 0)]] RWTexture2DArray<float4> storageImageArraysF[];
[[vk::binding(0, 0)]] RWTexture2D<float4> storageImagesF[];
[[vk::binding(4, 0)]] ConstantBuffer<UBO> uniformBuffers[];

static const int2 probe_offsets[] = {
    int2(0, 0),
    int2(1, 0),
    int2(0, 1),
    int2(1, 1)
};

float3 irradiance(float2 pos, PushConstant pc) {
    float2 base_coord = pos / pc.spacing - 0.5;
    int2 base_index = int2(floor(base_coord));
    float2 ratio = base_coord - float2(base_index);
    float4 weights = float4(
            (1.0 - ratio.x) * (1.0 - ratio.y),
            ratio.x * (1.0 - ratio.y),
            (1.0 - ratio.x) * ratio.y,
            ratio.x * ratio.y
    );
    int dir_count = int(pc.probe_size * pc.probe_size);

    float3 sum = 0.0;
    float weight_sum = 0.0;
    for (int i = 0; i < 4; i++)
    {
        int2 offset = probe_offsets[i];
        int2 index = base_index + offset;

        float2 coord = (float2(index) + 0.5) * pc.spacing;
        if (storageImagesF[pc.sdf_id].Load(int2(coord)).x < 0.1) continue;

        float3 sub_sum = 0.0;
        for (int j = 0; j < dir_count; j++)
        {
            int2 dir_index = int2(j % int(pc.probe_size), j / int(pc.probe_size));
            int2 texel = index * int(pc.probe_size) + dir_index;
            sub_sum += storageImageArraysF[pc.cascades_id].Load(int3(texel, 0)).rgb;
        }
        sum += weights[i] * sub_sum / dir_count;
        weight_sum += weights[i];
    }
    return weight_sum > 0.0 ? sum / weight_sum : 0.0;
}

[shader("compute")]
[numthreads(8, 8)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID, uniform PushConstant pc) {
    int2 local = int2(dispatchThreadID.xy);
    UBO ubo = uniformBuffers[0];
    if (any(local >= int2(ubo.scene_width, ubo.scene_height))) return;


    float2 center = float2(local) + 0.5;
    float3 emitted = irradiance(center, pc);
    bool inside = storageImagesF[pc.sdf_id].Load(int2(local)).x < 0.1;

    storageImagesF[pc.light_id].Store(local, inside ? float4(0.0) : float4(emitted, 1.0));
}