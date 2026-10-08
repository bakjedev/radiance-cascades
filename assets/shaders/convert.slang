#include "common.slang"

struct PushConstant {
    uint jfa_id;
}

[[vk::binding(0, 0)]] RWTexture2D<int4> storageImagesI[];
[[vk::binding(0, 0)]] RWTexture2D<uint4> storageImagesU[];
[[vk::binding(4, 0)]] ConstantBuffer<UBO> uniformBuffers[];

[shader("compute")]
[numthreads(8, 8)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID, uniform PushConstant pc) {
    int2 local = int2(dispatchThreadID.xy);
    UBO ubo = uniformBuffers[0];
    if (any(local >= int2(ubo.scene_width, ubo.scene_height))) return;

    uint drawn = storageImagesU[0].Load(local).x;

    if (drawn >= 1) {
        storageImagesI[pc.jfa_id].Store(local, int4(local, -1, -1));
        return;
    }
    storageImagesI[pc.jfa_id].Store(local, int4(-1, -1, local));
}