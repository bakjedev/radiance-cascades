[vk::constant_id(0)] const uint img_width = 0;
[vk::constant_id(1)] const uint img_height = 0;

struct PushConstant {
    uint jfa_id;
    uint sdf_id;
}

[[vk::binding(0, 0)]] RWTexture2D<float4> storageImagesF[];
[[vk::binding(0, 0)]] RWTexture2D<int4> storageImagesI[];

[shader("compute")]
[numthreads(8, 8)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID, uniform PushConstant pc) {
    int2 local = int2(dispatchThreadID.xy);
    if (any(local >= int2(img_width, img_height))) return;

    int2 seed = storageImagesI[pc.jfa_id].Load(local).xy;
    float dist = distance(float2(local), float2(seed));
    storageImagesF[pc.sdf_id].Store(local, float4(dist));
}