#include "common.slang"

struct PushConstant {
    uint composite_id;
    uint light_id;
    uint sdf_id;
    uint scene_col_id;
    uint scene_rad_id;
    float rim;
    uint light_level;
}

[[vk::binding(0, 0)]] RWTexture2D<float4> storageImagesF[];
[[vk::binding(4, 0)]] ConstantBuffer<UBO> uniformBuffers[];

[[vk::binding(2, 0)]] Texture2D textures[];

float3 light_bilinear(int id, float2 pos, int level, int2 size) {
    float2 base_coord = pos / float(1 << level) - 0.5;
    int2 base_idx = int2(floor(base_coord));
    float2 ratio = base_coord - float2(base_idx);

    float3 top_left = storageImagesF[id].Load(clamp(base_idx, 0, size - 1)).rgb;
    float3 top_right = storageImagesF[id].Load(clamp(base_idx + int2(1, 0), 0, size - 1)).rgb;
    float3 bottom_left = storageImagesF[id].Load(clamp(base_idx + int2(0, 1), 0, size - 1)).rgb;
    float3 bottom_right = storageImagesF[id].Load(clamp(base_idx + int2(1, 1), 0, size - 1)).rgb;

    return lerp(lerp(top_left, top_right, ratio.x), lerp(bottom_left, bottom_right, ratio.x), ratio.y);
}

[shader("compute")]
[numthreads(8, 8)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID, uniform PushConstant pc) {
    int2 local = int2(dispatchThreadID.xy);
    UBO ubo = uniformBuffers[0];
    int2 size = int2(ubo.scene_width, ubo.scene_height);
    if (any(local >= size)) return;

    float4 color = textures[pc.scene_col_id].Load(int3(local, 0));

    if (color.a == 0.0) {
        float3 light = storageImagesF[pc.light_id].Load(local).rgb;
        storageImagesF[pc.composite_id].Store(local, float4(light, 1.0));
        return;
    } else if (round(color.a * 255.0) == 2) { // sprite
        float3 light = light_bilinear(pc.light_id + pc.light_level, float2(local) + 0.5, pc.light_level, size >> pc.light_level);
        float3 radiance = textures[pc.scene_rad_id].Load(int3(local, 0)).xyz;
        float3 result = color.xyz * 2.0 * light + radiance;
        storageImagesF[pc.composite_id].Store(local, float4(result, 1.0));
        return;
    }

    float3 light = storageImagesF[pc.light_id].Load(local).rgb;

    float dist = storageImagesF[pc.sdf_id].Load(local).x;
    float3 radiance = textures[pc.scene_rad_id].Load(int3(local, 0)).xyz;

    float rim = exp(min(dist, 0.0) / pc.rim);
    if (abs(color.a - 0.5) < 0.002) rim = 1.0;
    float3 result = color.xyz * light * rim + radiance;

    storageImagesF[pc.composite_id].Store(local, float4(result, 1.0));
}