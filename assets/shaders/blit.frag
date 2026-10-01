#version 450
#extension GL_EXT_nonuniform_qualifier : require

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 outColor;

layout(push_constant) uniform PushConstant {
    uint composite_id;
} pc;

layout(binding = 2) uniform texture2D textures[];
layout(binding = 3) uniform sampler samplers[];

void main() {
    vec3 col = texture(sampler2D(textures[pc.composite_id], samplers[0]), uv).rgb;
    col = col / (1.0 + col); // reinhard
    outColor = vec4(col, 1.0);
}