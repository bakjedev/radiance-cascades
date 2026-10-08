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
    uint32_t size{50};
  } drawing;

  struct SceneSize {
    uint32_t width{1024};
    uint32_t height{1024};
  } scene_size;

  struct CascadesConfig {
    uint32_t cascades{6};
    float base_spacing{1.0f};
    float base_interval{90.0f};
    float base_length{0.8f};
  } cascades;
};

struct Material {
  float color[3];
  float padding{};
  float radiance[3];
  float paddington{};
};

class Renderer2D {
public:
  Renderer2D(Window& window, EventDispatcher& event_dispatcher, ResourceManager<ShaderResource>& resource_manager,
             FileSystem& file_system);
  ~Renderer2D();

  void render();

  void plot(const std::pair<float, float>& pos);

private:
  bool begin_frame();
  void run_frame();
  void end_frame();

  void compile();

  void draw_pass(vk::CommandBuffer cmd);
  void convert_pass(vk::CommandBuffer cmd);
  void jfa_pass(vk::CommandBuffer cmd, fwrk::ResourceID jfa_1, fwrk::ResourceID jfa_2);
  void sdf_pass(vk::CommandBuffer cmd);
  void cascades_pass(vk::CommandBuffer cmd, uint32_t cascade_width, uint32_t cascade_height, uint32_t probe_size,
                     float base_probe_spacing, uint32_t base_probe_dir_count, float base_probe_length);
  void merge_pass(vk::CommandBuffer cmd, uint32_t cascade_width, uint32_t cascade_height, uint32_t probe_size,
                  fwrk::ResourceID cascades);
  void composite_pass(vk::CommandBuffer cmd, float spacing, uint32_t probe_size);
  void blit_pass(vk::CommandBuffer cmd);
  void imgui_pass(vk::CommandBuffer cmd);

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

  vk::UniquePipelineLayout merge_pipeline_layout_;
  vk::UniqueShaderModule merge_shader_module_;
  vk::UniquePipeline merge_pipeline_;

  vk::UniquePipelineLayout composite_pipeline_layout_;
  vk::UniqueShaderModule composite_shader_module_;
  vk::UniquePipeline composite_pipeline_;

  vk::UniquePipelineLayout blit_pipeline_layout_;
  vk::UniqueShaderModule blit_shader_module_;
  vk::UniquePipeline blit_pipeline_;

  vk::UniqueDescriptorPool descriptor_pool_;
  vk::UniqueDescriptorSetLayout bindless_descriptor_set_layout_;
  vk::DescriptorSet bindless_descriptor_set_;

  vk::UniqueSampler sampler_;

  std::optional<Image> scene_image_;
  vk::UniqueImageView scene_image_view_;

  std::optional<Buffer> material_buffer_;
  std::optional<Buffer> ubo_;

  uint32_t current_frame_{};
  uint32_t image_index_{};

  FwrkAllocator fwrk_allocator_;
  fwrk::Context context_;
  std::vector<fwrk::ResourceID> swapchain_imports_;
  fwrk::ResourceID swapchain_proxy_;
  fwrk::ResourceID scene_image_import_;
  bool should_compile_ = true;

  std::vector<Material> materials_;
  uint8_t draw_material_ = 1;
  bool should_draw = false;
  std::pair<uint32_t, uint32_t> draw_pos_;

  void import_resources();

  void create_scene_image();

  static VkSurfaceKHR create_surface(const Window& window, const Instance& instance);
};
