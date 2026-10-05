[vk::constant_id(0)] const uint img_width = 0;
[vk::constant_id(1)] const uint img_height = 0;

struct PushConstant {
    uint read_id;
    uint write_id;
    uint k;
}

[[vk::binding(0, 0)]] RWTexture2D<int4> storageImagesI[];

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
    if (any(local >= int2(img_width, img_height))) return;

    int2 closest_seed = int2(-1);
    int closest_dist = 9999999;
    for (int i = 0; i < 9; i++) {
        int2 dir = directions[i] * int2(pc.k);
        int2 coord = local + dir;
        if (any(coord >= int2(img_width, img_height)) || any(coord < int2(0))) continue;
        int2 offset_point = storageImagesI[pc.read_id].Load(coord).xy;
        if (offset_point.x < 0) continue;
        int2 d = offset_point - local;
        int dist2 = d.x * d.x + d.y * d.y;
        if (dist2 < closest_dist) {
            closest_dist = dist2;
            closest_seed = offset_point;
        }
    }

    storageImagesI[pc.write_id].Store(local, int4(closest_seed, 0, 0));
}