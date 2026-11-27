#pragma once
#include <functional>
#include <glm/glm.hpp>
#include <vulkan/vulkan.hpp>

struct ImDrawData;

struct RendererCreateInfo {
  std::span<const char* const> instance_extensions;
  std::function<vk::SurfaceKHR(vk::Instance)> create_surface;
  vk::Extent2D initial_extent;
  std::function<std::vector<std::byte>(const std::string&)> load_shader;
};

struct RenderInfo {
  std::pair<uint32_t, uint32_t> window_size;
  ImDrawData* imgui_draw_data;
  glm::mat4 view{};
  glm::mat4 projection{};
};

struct GPUResource {
  uint32_t id;
};
struct GPUImage : GPUResource {};
struct GPUMaterial : GPUResource {};
struct GPUSampler : GPUResource {};

struct GeometryRange {
  uint32_t first_index;
  uint32_t index_count;
  int32_t vertex_offset;
};

enum class Filter : uint8_t {
  Nearest = 0,
  Linear,
  NearestMipMapNearest,
  LinearMipMapNearest,
  NearestMipMapLinear,
  LinearMipMapLinear,
};

enum class Wrap : uint8_t {
  ClampToEdge = 0,
  MirroredRepeat,
  Repeat,
};

struct MaterialInfo {
  glm::vec4 base_color;
  glm::vec3 emissive_color;
  float metallic, roughness, normal_scale, occlusion_scale;
  struct Textures {
    struct Texture {
      uint32_t index{UINT32_MAX};
      uint32_t tex_coord{UINT32_MAX};
      uint32_t sampler{UINT32_MAX};
    } albedo, metal_roughness, normal, occlusion, emissive;
  } textures;
  uint32_t alpha_mode;
  float alpha_cutoff;
};

struct Vertex {
  glm::vec3 position;
  glm::vec3 normal;
  glm::vec2 tex_coord_0;
  glm::vec2 tex_coord_1;
  glm::vec4 tangent;
};

struct GeometrySubmission {
  GeometryRange range;
  GPUMaterial material;
  glm::mat4 transform;
};

struct LightSubmission {
  glm::vec3 position;
  glm::vec3 radiance;
  float range;
};
struct PointLightSubmission : LightSubmission {};
struct SpotLightSubmission : LightSubmission {
  glm::vec3 direction;
  float cos_inner, cos_outer;
};
