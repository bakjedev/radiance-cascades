[vk::constant_id(0)] const uint img_width = 0;
[vk::constant_id(1)] const uint img_height = 0;

struct PushConstant {
    uint cascades_id;
    uint composite_id;
    float spacing;
    uint probe_size;
}

[[vk::binding(0, 0)]] RWTexture2DArray<float4> storageImageArraysF[];
[[vk::binding(0, 0)]] RWTexture2D<float4> storageImagesF[];

static const int2 probe_offsets[] = {
    int2(0, 0),
    int2(1, 0),
    int2(0, 1),
    int2(1, 1)
};

[shader("compute")]
[numthreads(8, 8)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID, uniform PushConstant pc) {
    int2 local = int2(dispatchThreadID.xy);
    if (any(local >= int2(img_width, img_height))) return;

    float2 base_coord = ((float2(local) + 0.5) / pc.spacing) - float2(0.5);
    int2 base_index = int2(floor(base_coord));
    float2 ratio = base_coord - float2(base_index);

    float4 weights = float4(
            (1.0 - ratio.x) * (1.0 - ratio.y),
            ratio.x * (1.0 - ratio.y),
            (1.0 - ratio.x) * ratio.y,
            ratio.x * ratio.y
    );

    int dir_count = int(pc.probe_size * pc.probe_size);

    float4 sum = float4(0.0);
    for (int i = 0; i < 4; i++)
    {
        float4 sub_sum = float4(0.0);
        for (int j = 0; j < dir_count; j++)
        {
            int2 offset = probe_offsets[i];
            int2 index = base_index + offset;
            int2 dir_index = int2(j % int(pc.probe_size), j / int(pc.probe_size));
            int2 texel = index * int(pc.probe_size) + dir_index;
            sub_sum += storageImageArraysF[pc.cascades_id].Load(int3(texel, 0)) * weights[i];
        }
        sum += sub_sum / dir_count;
    }

    storageImagesF[pc.composite_id].Store(local, sum);
}