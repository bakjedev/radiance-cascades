#include "../common.slang"

struct PushConstant {
    uint composite_id;
    uint frame_data_id;
    uint cascades_id;
    uint lights_data_id;
    uint shadow_id;
    uint shadow_sampler_id;
    uint lights_id;
    uint albedo_id;
    uint normal_id;
    uint pbr_id;
    uint emissive_id;
    uint depth_id;
}

struct FrameData {
    float4x4 inv_view_proj;
    float3 camera_pos;
    float3 sun_dir;
    float3 sun_color;
    float3 ambient;
    float3 ambient_indirect;
}

struct LightsData {
    float4x4 point_view_proj[4];
    float4x4 spot_view_proj[4];
    uint point_lights_base;
    uint spot_lights_base;
    uint point_light_count;
    uint spot_light_count;
}

struct Light {
    float3 position;
    float range;
    float3 direction;
    float angle_size;
    float3 radiance;
    float angle_offset;
}

struct Cascades {
    float4x4 world_to_shadow_clip[4];
    float4 world_units_per_texel;
    float4 filter_radius_per_texel;
    float inv_res;
    float blend;
}

struct Fragment {
    float3 normal;
    float3 dir;
    float3 color;
    float metallic;
    float roughness;
    float ao;
}

[[vk::binding(0, 0)]] RWTexture2D<float4> storageImagesF[];
[[vk::binding(1, 0)]] StructuredBuffer<Light> lightBuffers[];
[[vk::binding(2, 0)]] Texture2D textures[];
[[vk::binding(2, 0)]] Texture2DArray textureArrays[];
[[vk::binding(4, 0)]] ConstantBuffer<FrameData> frameDataUBOs[];
[[vk::binding(4, 0)]] ConstantBuffer<Cascades> cascadesUBOs[];
[[vk::binding(4, 0)]] ConstantBuffer<LightsData> lightsUBOs[];
[[vk::binding(3, 0)]] SamplerComparisonState shadowSamplers[];

static const uint SHADOW_CASCADE_COUNT = 4;
static const float NORMAL_OFFSET_TEXELS = 1.0;

static const float2 POISSON_DISK[16] = {
    float2(-0.94201624, -0.39906216), float2( 0.94558609, -0.76890725),
    float2(-0.09418410, -0.92938870), float2( 0.34495938,  0.29387760),
    float2(-0.91588581,  0.45771432), float2(-0.81544232, -0.87912464),
    float2(-0.38277543,  0.27676845), float2( 0.97484398,  0.75648379),
    float2( 0.44323325, -0.97511554), float2( 0.53742981, -0.47373420),
    float2(-0.26496911, -0.41893023), float2( 0.79197514,  0.19090188),
    float2(-0.24188840,  0.99706507), float2(-0.81409955,  0.91437590),
    float2( 0.19984126,  0.78641367), float2( 0.14383161, -0.14100790),
};

float3 oct_decode(float2 f)
{
    float3 n = float3(f, 1.0 - abs(f.x) - abs(f.y));
    float t = saturate(-n.z);
    n.xy += float2(n.x >= 0.0 ? -t : t, n.y >= 0.0 ? -t : t);
    return normalize(n);
}

float D_GGX(float NoH, float a) {
    float a2 = a * a;
    float d = (NoH * a2 - NoH) * NoH + 1.0;
    return a2 / (PI * d * d);
}

float V_SmithGGXCorrelated(float NoV, float NoL, float a) {
    float a2 = a * a;
    float gv = NoL * sqrt(NoV * NoV * (1.0 - a2) + a2);
    float gl = NoV * sqrt(NoL * NoL * (1.0 - a2) + a2);
    return 0.5 / (gv + gl);
}

float3 F_Schlick(float VoH, float3 f0) {
    float f = pow(1.0 - VoH, 5.0);
    return f + f0 * (1.0 - f);
}

float Fd_Lambert() { return 1.0 / PI; }

float3 evaluate_brdf(Fragment frag, float3 light) {
    float3 H  = normalize(frag.dir + light);
    float NoV = abs(dot(frag.normal, frag.dir)) + 1e-5;
    float NoL = saturate(dot(frag.normal, light));
    float NoH = saturate(dot(frag.normal, H));
    float VoH = saturate(dot(frag.dir, H));
    if (NoL <= 0.0) return 0.0;

    float a = frag.roughness * frag.roughness;
    float3 f0 = lerp(float3(0.04), frag.color, frag.metallic);

    float  D = D_GGX(NoH, a);
    float  V = V_SmithGGXCorrelated(NoV, NoL, a);
    float3 F = F_Schlick(VoH, f0);

    float3 Fr = D * V * F;
    float3 Fd = frag.color * (1.0 - frag.metallic) * Fd_Lambert();

    return (Fd + Fr) * NoL;
}

float distance_falloff(float distance_sq, float inv_range) {
    float norm_distance_sq = distance_sq * inv_range * inv_range;
    float window = saturate(1.0 - norm_distance_sq * norm_distance_sq);
    return (window * window) / max(distance_sq, 1e-4);
}

float3 evaluate_point_light(Fragment fragment, Light light, float3 world) {
    float3 distance = light.position - world;
    float  distance_sq = dot(distance, distance);
    float3 dir = distance * rsqrt(distance_sq);
    return evaluate_brdf(fragment, dir) * light.radiance * distance_falloff(distance_sq, 1.0 / light.range);
}

float3 evaluate_spot_light(Fragment fragment, Light light, float3 world) {
    float3 distance = light.position - world;
    float  distance_sq = dot(distance, distance);
    float3 dir = distance * rsqrt(distance_sq);

    float angle = saturate(dot(light.direction, -dir) * light.angle_size + light.angle_offset);
    angle *= angle;

    return evaluate_brdf(fragment, dir) * light.radiance * distance_falloff(distance_sq, 1.0 / light.range) * angle;
}

float interleaved_gradient_noise(float2 p) { return frac(52.9829189 * frac(dot(p, float2(0.06711056, 0.00583715)))); }

float pcf_poisson(Texture2DArray shadow_map, SamplerComparisonState shadow_sampler, float2 uv, float layer, float depth, float radius_uv, float2 pixel) {
    float rotation_angle = interleaved_gradient_noise(pixel) * 2.0 * PI;
    float sin_angle, cos_angle;
    sincos(rotation_angle, sin_angle, cos_angle);
    float2x2 rotation = float2x2(cos_angle, -sin_angle, sin_angle, cos_angle);

    float sum = 0.0;
    [unroll]
    for (uint i = 0; i < 16; ++i) {
        float2 offset = mul(rotation, POISSON_DISK[i]) * radius_uv;
        sum += shadow_map.SampleCmpLevelZero(shadow_sampler, float3(uv + offset, layer), depth);
    }
    return sum * (1.0 / 16.0);
}

bool project_cascade(Cascades cascades, uint idx, float3 world, float3 normal, float n_dot_l, out float3 shadow_coord, out float edge) {
    float normal_offset_distance = cascades.world_units_per_texel[idx] * NORMAL_OFFSET_TEXELS * (1.0 - n_dot_l);
    float3 shadow_ndc = mul(cascades.world_to_shadow_clip[idx], float4(world + normal * normal_offset_distance, 1.0)).xyz;
    float2 shadow_uv = shadow_ndc.xy * 0.5 + 0.5;

    float margin_uv = (cascades.filter_radius_per_texel[idx] + 1.0) * cascades.inv_res;
    float2 to_border = min(shadow_uv - margin_uv, (1.0 - margin_uv) - shadow_uv);
    edge = min(to_border.x, to_border.y) / (0.5 - margin_uv);
    shadow_coord = float3(shadow_uv, shadow_ndc.z);
    return edge >= 0.0 && shadow_ndc.z >= 0.0 && shadow_ndc.z <= 1.0;
}

float sample_cascade(Cascades cascades, Texture2DArray shadow_map, SamplerComparisonState shadow_sampler, uint idx, float3 coord, float2 pixel) {
    float radius_uv = cascades.filter_radius_per_texel[idx] * cascades.inv_res;
    return pcf_poisson(shadow_map, shadow_sampler, coord.xy, idx, coord.z, radius_uv, pixel);
}

float visibility(Cascades cascades, Texture2DArray shadow_map, SamplerComparisonState shadow_sampler, float3 world, float3 normal, float n_dot_l, float2 pixel) {
    for (uint i = 0; i < SHADOW_CASCADE_COUNT; i++) {
        float3 coord;
        float edge;
        if (!project_cascade(cascades, i, world, normal, n_dot_l, coord, edge)) continue;

        float shadow = sample_cascade(cascades, shadow_map, shadow_sampler, i, coord, pixel);
        if (edge >= cascades.blend) return shadow;

        float next_shadow = 1.0;
        if (i + 1 < SHADOW_CASCADE_COUNT) {
            float3 next_coord;
            float next_edge;
            if (!project_cascade(cascades, i + 1, world, normal, n_dot_l, next_coord, next_edge))
                return shadow;
            next_shadow = sample_cascade(cascades, shadow_map, shadow_sampler, i + 1, next_coord, pixel);
        }
        return lerp(next_shadow, shadow, smoothstep(0.0, 1.0, edge / cascades.blend));
    }
    return 1.0;
}

[shader("compute")]
[numthreads(8, 8)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID, uniform PushConstant pc) {
    int2 local = int2(dispatchThreadID.xy);
    RWTexture2D<float4> img = storageImagesF[pc.composite_id];
    uint2 size;
    img.GetDimensions(size.x, size.y);
    if (any(local >= size)) return;

    int3 pixel = int3(local, 0);

    float depth = textures[pc.depth_id].Load(pixel).x;

    if (depth == 0.0) {
        storageImagesF[pc.composite_id].Store(local, float4(0.02, 0.02, 0.02, 1.0));
        return;
    }

    FrameData frame_data = frameDataUBOs[pc.frame_data_id];
    float2 uv = (float2(local) + 0.5) / float2(size);
    float2 ndc_xy = uv * 2.0 - 1.0;
    float4 clip = float4(ndc_xy, depth, 1.0);
    float4 world = mul(frame_data.inv_view_proj, clip);
    world.xyz /= world.w;

    float4 albedo = textures[pc.albedo_id].Load(pixel);
    float2 normal_encoded = textures[pc.normal_id].Load(pixel).xy;
    float3 pbr = textures[pc.pbr_id].Load(pixel).xyz;
    float3 emissive = textures[pc.emissive_id].Load(pixel).xyz;

    Fragment fragment;
    fragment.normal = oct_decode(normal_encoded);
    fragment.dir = normalize(frame_data.camera_pos - world.xyz);
    fragment.color = albedo.rgb;
    fragment.metallic = pbr.r;
    fragment.roughness = clamp(pbr.g, 0.045, 1.0);
    fragment.ao = pbr.b;

    float3 color = 0.0;

    float n_dot_l = saturate(dot(fragment.normal, frame_data.sun_dir));
    float visibility = n_dot_l > 0.0 ? visibility(cascadesUBOs[pc.cascades_id],
                                                 textureArrays[pc.shadow_id],
                                                 shadowSamplers[pc.shadow_sampler_id],
                                                 world.xyz, fragment.normal, n_dot_l, float2(local)) : 0.0;

    color += evaluate_brdf(fragment, frame_data.sun_dir) * frame_data.sun_color * visibility;

    LightsData lights_data = lightsUBOs[pc.lights_data_id];
    for (uint i = 0; i < lights_data.point_light_count; ++i)
        color += evaluate_point_light(fragment, lightBuffers[pc.lights_id][lights_data.point_lights_base + i], world.xyz);
    for (uint i = 0; i < lights_data.spot_light_count; ++i)
        color += evaluate_spot_light(fragment, lightBuffers[pc.lights_id][lights_data.spot_lights_base + i], world.xyz);

    float3 ambient = lerp(frame_data.ambient_indirect, frame_data.ambient, fragment.normal.y * 0.5 + 0.5);
    color += ambient * fragment.color * (1.0 - fragment.metallic) * fragment.ao;

    color += emissive;

    img.Store(local, float4(color, 1.0));
    // img.Store(local, float4(fragment.normal, 1.0));
    // img.Store(local, float4(fragment.color, 1.0));
    // img.Store(local, float4(fragment.metallic, 0.0, 0.0, 1.0));
    // img.Store(local, float4(0.0, fragment.roughness, 0.0, 1.0));
    // img.Store(local, float4(fragment.ao.xxx, 1.0));
    // img.Store(local, float4(emissive, 1.0));
    // img.Store(local, float4(depth.xxx, 1.0));
    // img.Store(local, world);
    // img.Store(local, float4(visibility.xxx, 1.0));
}