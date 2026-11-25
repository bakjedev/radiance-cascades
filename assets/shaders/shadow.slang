struct VSInput {
    float3 pos : POSITION;
}

struct PushConstant {
    uint cascade;
    uint cascades_id;
    uint geometries_id;
}

struct Cascades {
    float4x4 world_to_shadow_clip[4];
    float4 world_units_per_texel;
}

struct Geometry {
    float4x4 transform;
    uint material;
}

[[vk::binding(1, 0)]] StructuredBuffer<Geometry> storageBuffersGeometry[];
[[vk::binding(4, 0)]] ConstantBuffer<Cascades> cascadesUBOs[];

[shader("vertex")]
float4 main(VSInput input, uint instanceID : SV_StartInstanceLocation, uniform PushConstant pc) : SV_Position{
    Geometry geometry = storageBuffersGeometry[pc.geometries_id][instanceID];
    float4 world = mul(geometry.transform,float4(input.pos, 1.0));
    return mul(cascadesUBOs[pc.cascades_id].world_to_shadow_clip[pc.cascade], world);
}
