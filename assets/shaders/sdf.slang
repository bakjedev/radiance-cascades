#include "common.slang"

struct PushConstant {
    uint jfa_id;
    uint sdf_id;
}

[[vk::binding(0, 0)]] RWTexture2D<float4> storageImagesF[];
[[vk::binding(0, 0)]] RWTexture2D<int4> storageImagesI[];
[[vk::binding(4, 0)]] ConstantBuffer<UBO> uniformBuffers[];

[shader("compute")]
[numthreads(8, 8)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID, uniform PushConstant pc) {
    int2 local = int2(dispatchThreadID.xy);
    UBO ubo = uniformBuffers[0];
    if (any(local >= int2(ubo.scene_width, ubo.scene_height))) return;

    int2 seed = storageImagesI[pc.jfa_id].Load(local).xy;
    float dist = distance(float2(local), float2(seed));
    storageImagesF[pc.sdf_id].Store(local, float4(dist));
}