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

static const uint AIR = 0;
static const uint TERRAIN = 1;
static const uint SPRITE = 2;