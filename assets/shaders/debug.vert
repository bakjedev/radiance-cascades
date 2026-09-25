#version 450

layout(location = 0) in vec3 aCol;
layout(location = 1) in vec2 aPos;

layout(constant_id = 0) const uint img_width = 0;
layout(constant_id = 1) const uint img_height = 0;

layout(location = 0) out vec3 color;

void main() {
    color = aCol;
    vec2 ndc = (aPos / vec2(img_width, img_height)) * 2.0 - 1.0;
    gl_Position = vec4(ndc, 0.0, 1.0);
}
