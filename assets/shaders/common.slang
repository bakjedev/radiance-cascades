static const float PI = 3.14159265359;
static const float EPSILON = 0.1;

struct UBO {
    uint32_t scene_width;
    uint32_t scene_height;
}

struct Material {
    float3 base_color;
    float3 radiance;
}
