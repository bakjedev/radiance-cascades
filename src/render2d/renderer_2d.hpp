#pragma once
#include "backend/device.hpp"
#include "backend/frame.hpp"
#include "backend/image.hpp"
#include "backend/instance.hpp"
#include "backend/swapchain.hpp"
#include "framework/context.hpp"
#include "fwrk_allocator.hpp"

class Window;
class EventDispatcher;

template<typename... Ts>
class ResourceManager;
struct ShaderResource;
class FileSystem;

constexpr uint32_t frames_in_flight = 2;

class Renderer2D {
public:
  Renderer2D(Window& window, ResourceManager<ShaderResource>& resource_manager, FileSystem& file_system);
  ~Renderer2D();

  void render();

private:
  bool begin_frame();
  void run_frame();
  void end_frame();

  Window& window_;
  ResourceManager<ShaderResource>& resource_manager_;
  FileSystem& file_system_;

  Instance instance_;
  Device device_;
  Swapchain swapchain_;
  std::array<Frame, frames_in_flight> frames_;
  std::vector<vk::UniqueSemaphore> submit_semaphores_;

  vk::UniquePipelineLayout pipeline_layout_;
  vk::UniqueShaderModule shader_module_;
  vk::UniquePipeline pipeline_;

  vk::UniqueDescriptorPool descriptor_pool_;
  vk::UniqueDescriptorSetLayout descriptor_set_layout_;
  vk::DescriptorSet descriptor_set_;

  std::optional<Image> scene_image_;
  vk::UniqueImageView scene_image_view_;

  uint32_t current_frame_{};
  uint32_t image_index_{};

  FwrkAllocator fwrk_allocator_;
  fwrk::Context context_;
  std::vector<fwrk::ResourceID> swapchain_imports_;
  fwrk::ResourceID swapchain_proxy_;
  fwrk::ResourceID scene_image_import_;
  bool should_compile_ = true;

  void import_resources();

  static VkSurfaceKHR create_surface(const Window& window, const Instance& instance);
};
