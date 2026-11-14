#pragma once
#include "framework/context.hpp"
#include "renderer_fwd.hpp"
#include "renderer_types.hpp"
#include "src/vulkan/device.hpp"
#include "src/vulkan/frame.hpp"
#include "src/vulkan/image.hpp"
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
  fwrk::Graph::FrameTimings get_timings();
  void request_recompile();
  static void imgui_new_frame();

  GPUImage create_texture(std::span<const uint8_t> data, const std::pair<uint32_t, uint32_t>& size);

protected:
  Instance instance_;
  Device device_;
  Swapchain swapchain_;
  std::array<Frame, frames_in_flight> frames_;
  std::vector<vk::UniqueSemaphore> submit_semaphores_;
  fwrk::Context context_;
  ImmediateSubmit immediate_;

private:
  vk::UniqueDescriptorPool descriptor_pool_;

protected:
  vk::UniqueDescriptorSetLayout bindless_descriptor_set_layout_;
  vk::DescriptorSet bindless_descriptor_set_;

  vk::UniqueSampler sampler_;

private:
  std::vector<fwrk::ResourceID> swapchain_imports_;
  uint32_t image_index_{};
  bool needs_recreate_ = false;

protected:
  fwrk::ResourceID swapchain_proxy_;

  uint32_t current_frame_{};

  bool should_compile_ = true;

  std::vector<std::pair<Image, vk::UniqueImageView>> texture_images_;

private:
  void import_swapchain_images();

  [[nodiscard]] fwrk::Context create_fwrk_context() const;
};
