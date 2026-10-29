#include "common.slang"

struct PushConstant {
    uint sdf_id;
    uint cascades_rw_id;
    uint cascades_id;
    uint scene_rad_id;
    uint cascade_width;
    uint cascade_height;
    uint cascade_idx;
    uint cascade_count;
    uint base_probe_size;
    float base_spacing;
    uint base_probe_dir_count;
    float base_length;
    uint light_id;
    uint scene_col_id;
    bool first;
}

[[vk::binding(0, 0)]] RWTexture2D<float4> storageImagesF[];
[[vk::binding(0, 0)]] RWTexture2DArray<float4> storageImageArraysF[];
[[vk::binding(1, 0)]] StructuredBuffer<Material> storageBuffers[];
[[vk::binding(2, 0)]] Texture2D textures[];
[[vk::binding(2, 0)]] Texture2DArray textureArrays[];
[[vk::binding(3, 0)]] SamplerState samplers[];
[[vk::binding(4, 0)]] ConstantBuffer<UBO> uniformBuffers[];

static const int2 probe_offsets[] = {
    int2(0, 0),
    int2(1, 0),
    int2(0, 1),
    int2(1, 1)
};

float4 trace(float2 start, float2 end, UBO ubo, PushConstant pc) {
    float2 delta = end - start;
    float2 dir = normalize(delta);
    float len = length(delta);

    float t = 0.0;
    for (int i = 0; i < 64; i++) {
        float2 point = start + dir * t;
        int2 coord = int2(floor(point));
        if (any(coord >= int2(ubo.scene_width, ubo.scene_height)) || any(coord < int2(0))) break;
        float distance = storageImagesF[pc.sdf_id].Load(coord).x;
        if (distance < EPSILON) {
            float3 radiance = textures[pc.scene_rad_id].Load(int3(coord, 0)).xyz;
            float4 result = float4(radiance, 0.0);
            if (!pc.first) {
                float3 color = textures[pc.scene_col_id].Load(int3(coord, 0)).xyz;
                result.xyz += color * storageImagesF[pc.light_id].Load(int2(floor(point - dir * 1.5))).xyz;
            }
            return result;
        }
        t += distance;
        if (t > len) break;
    }
    return float4(float3(0.0), 1.0);
}

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

//https://fgiesen.wordpress.com/2009/12/13/decoding-morton-codes/
uint part1by1(uint x) {
    x &= 0x0000FFFF;
    x = (x | (x << 8)) & 0x00FF00FF;
    x = (x | (x << 4)) & 0x0F0F0F0F;
    x = (x | (x << 2)) & 0x33333333;
    x = (x | (x << 1)) & 0x55555555;
    return x;
}

uint interval_to_dir(uint2 t) { return part1by1(t.x) | (part1by1(t.y) << 1); }

[shader("compute")]
[numthreads(8, 8)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID, uniform PushConstant pc) {
    int2 local = int2(dispatchThreadID.xy);
    if (any(local >= int2(pc.cascade_width, pc.cascade_height))) return;

    UBO ubo = uniformBuffers[0];

    int probe_size = int(pc.base_probe_size) << pc.cascade_idx;
    int2 probe = local / probe_size;
    float2 origin = (float2(probe) + 0.5) * (pc.base_spacing * float(1 << pc.cascade_idx));

    int2 interval = local % probe_size;
    int dir_index = int(interval_to_dir(uint2(interval)));

    float angle = 2.0 * PI * ((float(dir_index) + 0.5) / float(pc.base_probe_dir_count << (2 * pc.cascade_idx)));
    float2 dir = float2(cos(angle), sin(angle));

    float start_t = pc.base_length * ((pc.cascade_idx == 0) ? 0.0 : float(1 << (2 * pc.cascade_idx)));
    float end_t = pc.base_length * float(1 << (2 * (pc.cascade_idx + 1)));

    // at the last cascade, no cascade above this to merge with
    if (pc.cascade_idx >= pc.cascade_count - 1) {
        float4 result = trace(origin + dir * start_t, origin + dir * end_t, ubo, pc);
        storageImageArraysF[pc.cascades_rw_id].Store(int3(local, pc.cascade_idx), result);
        return;
    }

    float2 start = origin + dir * start_t;

    int neighbour_size = int(pc.base_probe_size) << (pc.cascade_idx + 1);
    float neighbour_spacing = pc.base_spacing * float(1 << (pc.cascade_idx + 1));
    float neighbour_start_t = pc.base_length * float(1 << (2 * (pc.cascade_idx + 1)));

    float4 weights = float4(0.0);
    int2 base_index = get_base_index(origin, float2(neighbour_spacing), weights);

    float2 inv_size = 1.0 / float2(pc.cascade_width, pc.cascade_height);
    float2 dir_tile_center = float2(2 * interval + 1);

    float4 merged = float4(0.0);
    for (int i = 0; i < 4; i++) {
        int2 index = base_index + probe_offsets[i];
        float2 pos = (float2(index) + 0.5) * neighbour_spacing;

        float2 end = pos + dir * neighbour_start_t;

        float4 result = trace(start, end, ubo, pc);

        // hardware bilinear sampling the 4 directions, because of morton layout the 4 neighbours are now exactly under each texel in the next cascade
        float2 point = float2(index * neighbour_size) + dir_tile_center;
        float4 radiance = textureArrays[pc.cascades_id].SampleLevel(samplers[1], float3(point * inv_size, float(pc.cascade_idx + 1)), 0);

        merged += merge(result, radiance) * weights[i];
    }

    storageImageArraysF[pc.cascades_rw_id].Store(int3(local, pc.cascade_idx), merged);
}