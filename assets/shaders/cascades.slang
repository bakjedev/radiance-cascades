static const float PI = 3.14159265359;

[vk::constant_id(0)] const uint img_width = 0;
[vk::constant_id(1)] const uint img_height = 0;

struct PushConstant {
    uint sdf_id;
    uint cascades_id;
    uint cascade_width;
    uint cascade_height;
    uint cascade;
    uint base_probe_size;
    float base_spacing;
    uint base_probe_dir_count;
    float base_length;
}

[[vk::binding(0, 0)]] RWTexture2D<float4> storageImagesF[];
[[vk::binding(0, 0)]] RWTexture2D<uint4> storageImagesU[];
[[vk::binding(0, 0)]] RWTexture2DArray<float4> storageImageArraysF[];

struct Material {
    float3 base_color;
    float3 radiance;
}

[[vk::binding(1, 0)]] StructuredBuffer<Material> storageBuffers[];

[shader("compute")]
[numthreads(8, 8)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID, uniform PushConstant pc) {
    int2 local = int2(dispatchThreadID.xy);
    if (any(local >= int2(pc.cascade_width, pc.cascade_height))) return;

    int probe_size = int(pc.base_probe_size) << pc.cascade;
    int2 probe = local / probe_size;
    float2 origin = (float2(probe) + 0.5) * (pc.base_spacing * float(1 << pc.cascade));

    int2 interval = local % probe_size;
    int dir_index = interval.y * probe_size + interval.x;

    float angle = 2.0 * PI * ((float(dir_index) + 0.5) / float(pc.base_probe_dir_count << (2 * pc.cascade)));
    float2 dir = float2(cos(angle), sin(angle));

    float start_t = pc.base_length * ((pc.cascade == 0) ? 0.0 : float(1 << (2 * pc.cascade)));
    float end_t = pc.base_length * float(1 << (2 * (pc.cascade + 1)));

    float t = start_t;
    bool hit = false;
    uint8_t hit_mat;
    for (int i = 0; i < 64; i++) {
        float2 point = origin + dir * t;
        int2 coord = int2(floor(point));
        if (any(coord >= int2(img_width, img_height)) || any(coord < int2(0))) break;
        float distance = storageImagesF[pc.sdf_id].Load(coord).x;
        if (distance < 0.1) {
            hit = true;
            hit_mat = uint8_t(storageImagesU[0].Load(coord).x);
            break;
        };
        t += distance;
        if (t > end_t) break;
    }


    float3 radiance = 0.0;
    float alpha = 1.0;
    if (hit) {
        radiance = storageBuffers[0][hit_mat].radiance;
        alpha = 0.0;
    }
    storageImageArraysF[pc.cascades_id].Store(int3(local, pc.cascade), float4(radiance, alpha));
}