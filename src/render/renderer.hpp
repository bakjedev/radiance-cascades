#pragma once
#include <functional>
#include <span>
#include <string>
#include <vulkan/vulkan_core.h>

#include "framework/graph.hpp"

struct ImDrawData;

struct RendererCreateInfo {
  std::span<const char* const> instance_extensions;
  std::function<VkSurfaceKHR(VkInstance)> create_surface;
  VkExtent2D initial_extent;
  std::function<std::vector<std::byte>(const std::string&)> load_shader;
};

struct RenderInfo {
  std::pair<uint32_t, uint32_t> window_size;
  ImDrawData* imgui_draw_data;
};

template<class T>
concept Renderer = std::constructible_from<T, const RendererCreateInfo&> && requires(T renderer, RenderInfo info) {
  renderer.render(info);
  renderer.request_recompile();
  { renderer.get_timings() } -> std::same_as<fwrk::Graph::FrameTimings>;
  T::imgui_new_frame();
};
