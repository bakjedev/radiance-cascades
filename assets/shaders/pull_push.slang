#include "common.slang"

struct PushConstant {
    uint read_id;
    uint write_id;
    uint width;
    uint height;
}

[[vk::binding(0, 0)]] RWTexture2D<float4> storageImagesF[];

// distances from fine to coarse are 1.5, 0.5, 0.5, 1.5
// distance to next coarse is 2 so 2 - 1.5, 2 - 0.5, etc.
// so 0.5, 1.5, 1.5, 0.5
// to make simpler double to create whole numbers.
static const float h[4] = float[](1.0, 3.0, 3.0, 1.0);

[shader("compute")]
[numthreads(8, 8)]
void pull(uint3 dispatchThreadID : SV_DispatchThreadID, uniform PushConstant pc) {
    int2 local = int2(dispatchThreadID.xy);
    if (any(local >= int2(pc.width, pc.height))) return;
    float4 sum = float4(0.0);
    int2 base = 2 * local - 1;
    for (int j = 0; j < 4; j++) {
        for (int i = 0; i < 4; i++) {
            int2 coord = base + int2(i, j);
            if (any(coord < 0) || any(coord >= int2(pc.width << 1, pc.height << 1))) continue;
            sum += h[i] * h[j] * storageImagesF[pc.read_id].Load(coord);
        }
    }

    float3 x = sum.a > 0.0 ? sum.rgb / sum.a : float3(0);
    // if atleast one of the most important pixels is confident then its 100% confident. We don't care about more than 100% confidence.
    float w = min(sum.a / 9.0, 1.0);
    
    storageImagesF[pc.write_id].Store(local, float4(x * w, w));
}

[shader("compute")]
[numthreads(8, 8)]
void push(uint3 dispatchThreadID : SV_DispatchThreadID, uniform PushConstant pc) {
    int2 local = int2(dispatchThreadID.xy);
    if (any(local >= int2(pc.width, pc.height))) return;
    int2 coarse = local >> 1;
    int2 dir = int2((local.x & 1) == 0 ? -1 : 1, (local.y & 1) == 0 ? -1 : 1);
    int2 coarse_size = int2(pc.width >> 1, pc.height >> 1) - 1;
    float3 upsampled =
        (9.0/16.0) * storageImagesF[pc.read_id].Load(clamp(coarse, int2(0), coarse_size)).rgb +
        (3.0/16.0) * storageImagesF[pc.read_id].Load(clamp(coarse + int2(dir.x, 0), int2(0), coarse_size)).rgb +
        (3.0/16.0) * storageImagesF[pc.read_id].Load(clamp(coarse + int2(0, dir.y), int2(0), coarse_size)).rgb +
        (1.0/16.0) * storageImagesF[pc.read_id].Load(clamp(coarse + dir, int2(0), coarse_size)).rgb;
    float4 fine = storageImagesF[pc.write_id].Load(local);

    float3 filled = fine.rgb + upsampled * (1.0 - fine.a);
    storageImagesF[pc.write_id].Store(local, float4(filled, 1.0));
}