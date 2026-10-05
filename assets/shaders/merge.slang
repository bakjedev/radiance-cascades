struct PushConstant {
    uint cascades_id;
    uint cascade;
    uint cascade_width;
    uint cascade_height;
    uint base_probe_size;
}

[[vk::binding(0, 0)]] RWTexture2DArray<float4> storageImageArraysF[];

static const int2 probe_offsets[] = {
    int2(0, 0),
    int2(1, 0),
    int2(0, 1),
    int2(1, 1)
};

float4 get_weights(float2 ratio) {
    return float4(
            (1.0 - ratio.x) * (1.0 - ratio.y),
            ratio.x * (1.0 - ratio.y),
            (1.0 - ratio.x) * ratio.y,
            ratio.x * ratio.y
    );
}

int2 get_base_index(float2 center, float2 target_size, out float4 weights) {
    float2 tl_coord = (center / target_size) - float2(0.5);
    int2 base_index = int2(floor(tl_coord));
    weights = get_weights(tl_coord - float2(base_index));
    return base_index;
}

float4 merge(float4 n, float4 n1) {
    const float3 radiance = n.rgb + (n1.rgb * n.a);
    return float4(radiance, n.a * n1.a);
}

[shader("compute")]
[numthreads(8, 8)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID, uniform PushConstant pc) {
    int2 local = int2(dispatchThreadID.xy);
    if (any(local >= int2(pc.cascade_width, pc.cascade_height))) return;

    int probe_size = int(pc.base_probe_size) << pc.cascade;
    int2 probe = local / probe_size;

    int2 dir = local % probe_size;
    int dir_index = dir.y * probe_size + dir.x;

    float2 probe_center = (float2(probe) + 0.5) * float(probe_size);
    int probe_size_n1 = int(pc.base_probe_size) << (pc.cascade + 1);

    float4 weights = float4(0.0);
    int2 base_index = get_base_index(probe_center, float2(probe_size_n1), weights);

    int base_dir_index = dir_index * 4;

    float4 n = storageImageArraysF[pc.cascades_id].Load(int3(local, pc.cascade));

    float4 merged = float4(0.0);
    for (int d = 0; d < 4; d++) {
        float4 radiance = float4(0.0);
        for (int p = 0; p < 4; p++) {
            int2 offset = probe_offsets[p];
            int2 index = base_index + offset;
            int dir_index = base_dir_index + d;
            int2 dir_coord = int2(
                    dir_index % probe_size_n1,
                    dir_index / probe_size_n1
            );

            int2 texel = index * probe_size_n1 + dir_coord;
            float4 n1 = storageImageArraysF[pc.cascades_id].Load(int3(texel, pc.cascade + 1));

            radiance += merge(n, n1) * weights[p];
        }
        merged += radiance / 4.0;
    }

    storageImageArraysF[pc.cascades_id].Store(int3(local, pc.cascade), merged);
}