struct VSOutput {
    float4 pos : SV_Position;
    float2 uv : TEXCOORD0;
}

// https://www.saschawillems.de/blog/2016/08/13/vulkan-tutorial-on-rendering-a-fullscreen-quad-without-buffers/
[shader("vertex")]
VSOutput vs_main(uint vertexID : SV_VulkanVertexID) {
    VSOutput out;
    out.uv = float2((vertexID << 1) & 2, vertexID & 2);
    out.pos = float4(out.uv * 2.0 - 1.0, 0.0, 1.0);
    return out;
}

struct PushConstant {
    uint composite_id;
}

[[vk::binding(2, 0)]] Texture2D textures[];
[[vk::binding(3, 0)]] SamplerState samplers[];

[shader("fragment")]
float4 fs_main(VSOutput input, uniform PushConstant pc) : SV_Target {
    float3 col = textures[pc.composite_id].Sample(samplers[0], input.uv).xyz;
    col = col / (1.0 + col); // reinhard
    return float4(col, 1.0);
}