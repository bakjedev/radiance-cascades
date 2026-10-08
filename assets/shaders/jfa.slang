#include "common.slang"

struct PushConstant {
    uint read_id;
    uint write_id;
    uint k;
}

[[vk::binding(0, 0)]] RWTexture2D<int4> storageImagesI[];
[[vk::binding(4, 0)]] ConstantBuffer<UBO> uniformBuffers[];

static const int2 directions[] = {
    int2(0, 0),
    int2(0, 1),
    int2(1, 1),
    int2(1, 0),
    int2(1, -1),
    int2(0, -1),
    int2(-1, -1),
    int2(-1, 0),
    int2(-1, 1)
};

[shader("compute")]
[numthreads(8, 8)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID, uniform PushConstant pc) {
    int2 local = int2(dispatchThreadID.xy);
    UBO ubo = uniformBuffers[0];
    if (any(local >= int2(ubo.scene_width, ubo.scene_height))) return;

    int2 closest_seed = int2(-1);
    int2 closest_seed_empty = int2(-1);
    int closest_dist = int::maxValue;
    int closest_dist_empty = int::maxValue;
    for (int i = 0; i < 9; i++) {
        int2 dir = directions[i] * int2(pc.k);
        int2 coord = local + dir;
        if (any(coord >= int2(ubo.scene_width, ubo.scene_height)) || any(coord < int2(0))) continue;
        int4 offset_point = storageImagesI[pc.read_id].Load(coord);
        if (offset_point.x >= 0) {
            int2 d = offset_point.xy - local;
            int dist2 = d.x * d.x + d.y * d.y;
            if (dist2 < closest_dist) {
                closest_dist = dist2;
                closest_seed = offset_point.xy;
            }
        }
        if (offset_point.z >= 0) {
            int2 d = offset_point.zw - local;
            int dist2 = d.x * d.x + d.y * d.y;
            if (dist2 < closest_dist_empty) {
                closest_dist_empty = dist2;
                closest_seed_empty = offset_point.zw;
            }
        }
    }

    storageImagesI[pc.write_id].Store(local, int4(closest_seed, closest_seed_empty));
}