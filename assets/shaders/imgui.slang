// backends/vulkan/glsl_shader.vert, compiled with:
// # glslangValidator -V -x -o glsl_shader.vert.u32 glsl_shader.vert
/*
#version 450 core
layout(location = 0) in vec2 aPos;
layout(location = 1) in vec2 aUV;
layout(location = 2) in vec4 aColor;
layout(push_constant) uniform uPushConstant { vec2 uScale; vec2 uTranslate; } pc;

out gl_PerVertex { vec4 gl_Position; };
layout(location = 0) out struct { vec4 Color; vec2 UV; } Out;

void main()
{
    Out.Color = aColor;
    Out.UV = aUV;
    gl_Position = vec4(aPos * pc.uScale + pc.uTranslate, 0, 1);
}

translated to slang + srgb to linear conversion
*/

struct VSInput {
    float2 pos : POSITION;
    float2 uv : TEXCOORD;
    float4 col : COLOR0;
};

struct VSOutput {
    float4 pos : SV_Position;
    float4 col : COLOR0;
    float2 uv : TEXCOORD0;
}

struct PushConstant {
    float2 scale;
    float2 translate;
}

float3 srgb_to_linear(float3 c) {
    return select(c <= 0.04045, c / 12.92, pow((c + 0.055) / 1.055, 2.4));
}

[shader("vertex")]
VSOutput main(VSInput in, uniform PushConstant pc)
{
    VSOutput out;
    out.pos = float4(in.pos * pc.scale + pc.translate, 0.0, 1.0);
    out.uv = in.uv;
    out.col = float4(srgb_to_linear(in.col.rgb), in.col.a);
    return out;
}