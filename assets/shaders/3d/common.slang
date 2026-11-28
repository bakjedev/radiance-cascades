static const float PI = 3.14159265359;

struct FrameData {
    float4x4 inv_view_proj;
    float3 camera_pos;
    float3 sun_dir;
    float3 sun_color;
    float3 ambient;
    float3 ambient_indirect;
}

struct Geometry {
    float4x4 transform;
    float4 bounds;
    uint primitive;
    uint material;
    uint flags;
}

struct Cascades {
    float4x4 world_to_shadow_clip[4];
    float4 world_units_per_texel;
    float4 filter_radius_per_texel;
    float inv_res;
    float blend;
}