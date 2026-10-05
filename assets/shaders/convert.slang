[vk::constant_id(0)] const uint img_width = 0;
[vk::constant_id(1)] const uint img_height = 0;

struct PushConstant {
    uint jfa_id;
}

[[vk::binding(0, 0)]] RWTexture2D<int4> storageImagesI[];
[[vk::binding(0, 0)]] RWTexture2D<uint4> storageImagesU[];

[shader("compute")]
[numthreads(8, 8)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID, uniform PushConstant pc) {
    int2 local = int2(dispatchThreadID.xy);
    if (any(local >= int2(img_width, img_height))) return;

    uint drawn = storageImagesU[0].Load(local).x;

    if (drawn >= 1) {
        storageImagesI[pc.jfa_id].Store(local, int4(local, 0, 0));
        return;
    }
    storageImagesI[pc.jfa_id].Store(local, int4(-1, -1, 0, 0));
}