struct VSInput {
    float3 col : TEXCOORD;
    float2 pos : POSITION;
};

[vk::constant_id(0)] const uint img_width = 0;
[vk::constant_id(1)] const uint img_height = 0;

struct VSOutput {
    float4 pos : SV_Position;
    float3 col : COLOR;
}

[shader("vertex")]
VSOutput vs_main(VSInput in) {
    VSOutput out;
    out.col = in.col;
    float2 ndc = (in.pos / float2(img_width, img_height)) * 2.0 - 1.0;
    out.pos = float4(ndc, 0.0, 1.0);
    return out;
}

[shader("fragment")]
float4 fs_main(VSOutput in) : SV_Target {
    return float4(in.col, 1.0);
}