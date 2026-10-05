[vk::constant_id(0)] const uint img_width = 0;
[vk::constant_id(1)] const uint img_height = 0;

struct PushConstant {
    int2 pos;
    uint size;
    uint8_t material_id;
}

[[vk::binding(0, 0)]] RWTexture2D<uint4> storageImagesU[];

[shader("compute")]
[numthreads(8, 8)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID, uniform PushConstant pc) {
    int2 local = int2(dispatchThreadID.xy);
    if (any(local >= int2(pc.size))) return;
    int2 start = pc.pos - int2(pc.size / 2);
    int2 coord = start + local;
    if (any(coord >= int2(img_width, img_height)) || any(coord < int2(0))) return;

    float dist = length(float2(coord) - float2(pc.pos));
    float radius = float(pc.size) / 2.0;
    if (dist >= radius) {
        return;
    }

    storageImagesU[0].Store(coord, int4(pc.material_id));
}