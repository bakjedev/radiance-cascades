struct VSOutput {
    float4 pos : SV_Position;
    float2 uv : TEXCOORD0;
}

float3 aces(float3 x)
{
    float a = 2.51f;
    float b = 0.03f;
    float c = 2.43f;
    float d = 0.59f;
    float e = 0.14f;
    return saturate((x*(a*x+b))/(x*(c*x+d)+e));
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
    //col = col / (1.0 + col); // reinhard
    col = aces(col);
    return float4(col, 1.0);
}