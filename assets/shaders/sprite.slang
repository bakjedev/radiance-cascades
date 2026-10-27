#include "common.slang"

struct Sprite {
    float3 tint;
    uint texture;
    float2 pos;
    float2 size;
}

struct VSOutput {
    float4 pos : SV_Position;
    float2 uv : TEXCOORD0;
    float3 tint: COLOR0;
    nointerpolation uint tex;
}

struct FSOutput {
    float4 color : SV_Target0;
    float3 radiance : SV_Target1;
    int4 jfa : SV_Target2;
}

struct PushConstant {
    uint sprite_textures_id;
    uint sprite_buffer_id;
}

[[vk::binding(4, 0)]] ConstantBuffer<UBO> uniformBuffers[];
[[vk::binding(1, 0)]] StructuredBuffer<Sprite> storageBuffers[];
[[vk::binding(2, 0)]] Texture2D textures[];
[[vk::binding(3, 0)]] SamplerState samplers[];

[shader("vertex")]
VSOutput vs_main(uint vertexID : SV_VulkanVertexID, uint instanceID : SV_VulkanInstanceID, uniform PushConstant pc) {
    Sprite sprite = storageBuffers[pc.sprite_buffer_id][instanceID];

    uint w, h;
    textures[pc.sprite_textures_id + sprite.texture].GetDimensions(w, h);

    float2 uv = float2(vertexID & 1, vertexID >> 1);
    float2 size = float2(w, h) * sprite.size;
    float2 px = sprite.pos - size / 2 + uv * size;

    UBO ubo = uniformBuffers[0];
    float2 inv_screen_size = float2(1.0 / float(ubo.scene_width), 1.0 / float(ubo.scene_height));

    VSOutput out;
    out.pos = float4(px * inv_screen_size * 2.0 - 1.0, 0.0, 1.0);
    out.uv = uv;
    out.tint = sprite.tint;
    out.tex = pc.sprite_textures_id + sprite.texture;
    return out;
}

[shader("fragment")]
FSOutput fs_main(VSOutput input) {
    FSOutput out;
    float4 color = textures[input.tex].Sample(samplers[0], input.uv);
    if (color.a == 0.0) discard;
    out.color = float4(color.xyz * input.tint, 2.0 / 255.0);
    out.radiance = 0.0;
    out.jfa = int4(int2(input.pos.xy), -1, -1);
    return out;
}