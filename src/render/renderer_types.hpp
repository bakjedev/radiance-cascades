#pragma once
#include <functional>
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
