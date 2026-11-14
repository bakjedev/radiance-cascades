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
      uint32_t index;
      uint32_t tex_coord;
    } albedo, metal_roughness, normal, occlusion, emissive;
  } textures;
};
