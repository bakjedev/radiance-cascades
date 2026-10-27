#include "common.slang"

struct VSOutput {
    float4 pos : SV_Position;
}

struct FSOutput {
    float4 color : SV_Target0;
    float3 radiance : SV_Target1;
    int4 jfa : SV_Target2;
}

[[vk::binding(2, 0)]] Texture2D textures[];
[[vk::binding(2, 0)]] Texture2D<uint> texturesU[];
[[vk::binding(1, 0)]] StructuredBuffer<Material> storageBuffers[];

// https://www.saschawillems.de/blog/2016/08/13/vulkan-tutorial-on-rendering-a-fullscreen-quad-without-buffers/
[shader("vertex")]
VSOutput vs_main(uint vertexID : SV_VulkanVertexID) {
    VSOutput out;
    float2 uv = float2((vertexID << 1) & 2, vertexID & 2);
    out.pos = float4(uv * 2.0 - 1.0, 0.0, 1.0);
    return out;
}

struct PushConstant {
    uint draw_id;
}

[shader("fragment")]
FSOutput fs_main(VSOutput input, uniform PushConstant pc) {
    FSOutput out;

    int2 coord = (int2)input.pos.xy;
    uint mat_id = texturesU[pc.draw_id].Load(int3(coord, 0));
    Material mat = storageBuffers[0][mat_id];

    bool solid = mat_id > 0;

    out.color = float4(mat.base_color, solid ? 1.0 / 255.0 : 0.0);
    out.radiance = mat.radiance;
    out.jfa = solid ? int4(coord, -1, -1) : int4(-1, -1, coord);

    return out;
}