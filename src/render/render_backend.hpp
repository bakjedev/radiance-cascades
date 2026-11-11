#pragma once
#include "framework/context.hpp"
#include "renderer.hpp"
#include "src/vulkan/device.hpp"
#include "src/vulkan/frame.hpp"
#include "src/vulkan/immediate_submit.hpp"
#include "src/vulkan/instance.hpp"
#include "src/vulkan/swapchain.hpp"

class RenderBackend {
public:
  static constexpr uint32_t frames_in_flight = 2;

  explicit RenderBackend(const RendererCreateInfo& create_info);
  ~RenderBackend();
  bool begin_frame(vk::Extent2D extent);
  void execute_frame();
  void end_frame();
  void window_resize(int width, int height);
  fwrk::Context& framework();
  Device& device();
  ImmediateSubmit& immediate();
  [[nodiscard]] const Frame& get_current_frame() const;
  [[nodiscard]] vk::Format swapchain_format() const;
  [[nodiscard]] fwrk::ResourceID swapchain_proxy() const;
  [[nodiscard]] const vk::DescriptorSetLayout& bindless_layout() const;
  [[nodiscard]] const vk::DescriptorSet& bindless_set() const;
  [[nodiscard]] vk::Sampler sampler() const;
  fwrk::Graph::FrameTimings get_timings();
  void request_recompile();
  [[nodiscard]] bool should_compile() const;
  static void imgui_new_frame();
  [[nodiscard]] uint32_t frame_index() const;

private:
  Instance instance_;
  Device device_;
  Swapchain swapchain_;
  std::array<Frame, frames_in_flight> frames_;
  std::vector<vk::UniqueSemaphore> submit_semaphores_;
  fwrk::Context context_;
  ImmediateSubmit immediate_;

  vk::UniqueDescriptorPool descriptor_pool_;
  vk::UniqueDescriptorSetLayout bindless_descriptor_set_layout_;
  vk::DescriptorSet bindless_descriptor_set_;

  vk::UniqueSampler sampler_;

  std::vector<fwrk::ResourceID> swapchain_imports_;
  fwrk::ResourceID swapchain_proxy_;

  uint32_t current_frame_{};
  uint32_t image_index_{};

  bool needs_recreate_ = false;
  bool should_compile_ = true;

  void import_swapchain_images();

  [[nodiscard]] fwrk::Context create_fwrk_context() const;
};
