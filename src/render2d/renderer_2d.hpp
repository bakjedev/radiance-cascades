#pragma once
#include "backend/buffer.hpp"
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

struct Renderer2DConfig {
  struct DrawConfig {
    uint32_t size{5};
  } drawing;

  struct SceneSize {
    uint32_t width{256};
    uint32_t height{256};
  } scene_size;

  struct CascadesConfig {
    uint32_t cascades{4};
    float base_spacing{8.0f};
    float base_interval{90.0f};
    float base_length{1.0f};
  } cascades;
};

class Renderer2D {
public:
  Renderer2D(Window& window, EventDispatcher& event_dispatcher, ResourceManager<ShaderResource>& resource_manager,
             FileSystem& file_system);
  ~Renderer2D();

  void render();

  void plot(const std::pair<float, float>& pos, uint8_t material);

private:
  bool begin_frame();
  void run_frame();
  void end_frame();

  Window& window_;
  EventDispatcher& event_dispatcher_;
  ResourceManager<ShaderResource>& resource_manager_;
  FileSystem& file_system_;

  Renderer2DConfig config_;

  Instance instance_;
  Device device_;
  Swapchain swapchain_;
  std::array<Frame, frames_in_flight> frames_;
  std::vector<vk::UniqueSemaphore> submit_semaphores_;

  vk::UniquePipelineLayout draw_pipeline_layout_;
  vk::UniqueShaderModule draw_shader_module_;
  vk::UniquePipeline draw_pipeline_;

  vk::UniquePipelineLayout convert_pipeline_layout_;
  vk::UniqueShaderModule convert_shader_module_;
  vk::UniquePipeline convert_pipeline_;

  vk::UniquePipelineLayout jfa_pipeline_layout_;
  vk::UniqueShaderModule jfa_shader_module_;
  vk::UniquePipeline jfa_pipeline_;

  vk::UniquePipelineLayout sdf_pipeline_layout_;
  vk::UniqueShaderModule sdf_shader_module_;
  vk::UniquePipeline sdf_pipeline_;

  vk::UniquePipelineLayout cascades_pipeline_layout_;
  vk::UniqueShaderModule cascades_shader_module_;
  vk::UniquePipeline cascades_pipeline_;

  vk::UniqueDescriptorPool descriptor_pool_;
  vk::UniqueDescriptorSetLayout bindless_descriptor_set_layout_;
  vk::DescriptorSet bindless_descriptor_set_;

  std::optional<Image> scene_image_;
  vk::UniqueImageView scene_image_view_;

  std::optional<Buffer> material_buffer_;

  uint32_t current_frame_{};
  uint32_t image_index_{};

  FwrkAllocator fwrk_allocator_;
  fwrk::Context context_;
  std::vector<fwrk::ResourceID> swapchain_imports_;
  fwrk::ResourceID swapchain_proxy_;
  fwrk::ResourceID scene_image_import_;
  bool should_compile_ = true;

  uint8_t draw_material_ = 0;
  std::pair<uint32_t, uint32_t> draw_pos_;

  void import_resources();

  static VkSurfaceKHR create_surface(const Window& window, const Instance& instance);
};
