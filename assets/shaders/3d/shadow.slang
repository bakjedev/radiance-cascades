#include "common.slang"

struct VSInput {
    float3 pos : POSITION;
}

struct PushConstant {
    uint cascade;
    uint cascades_id;
    uint geometries_id;
}

[[vk::binding(1, 0)]] StructuredBuffer<Geometry> storageBuffersGeometry[];
[[vk::binding(4, 0)]] ConstantBuffer<Cascades> cascadesUBOs[];

[shader("vertex")]
float4 main(VSInput input, uint instanceID : SV_StartInstanceLocation, uniform PushConstant pc) : SV_Position{
    Geometry geometry = storageBuffersGeometry[pc.geometries_id][instanceID];
    float4 world = mul(geometry.transform,float4(input.pos, 1.0));
    return mul(cascadesUBOs[pc.cascades_id].world_to_shadow_clip[pc.cascade], world);
}
