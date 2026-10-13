#pragma once
#include "backend/buffer.hpp"
#include "backend/device.hpp"
#include "backend/frame.hpp"
#include "backend/image.hpp"
#include "backend/instance.hpp"
#include "backend/swapchain.hpp"
#include "framework/context.hpp"

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

  struct CompositeConfig {
    float rim{12.0f};
  } composite;
};

struct Material {
  float color[3];
  float padding{};
  float radiance[3];
  float paddington{};
};

class EmptyAllocator : public fwrk::Allocator {
public:
  std::optional<fwrk::PhysicalImage> create_image(const fwrk::ImageCreateInfo&) override { return std::nullopt; }
  std::optional<fwrk::PhysicalBuffer> create_buffer(const fwrk::BufferCreateInfo&) override { return std::nullopt; }
  void destroy_image(fwrk::PhysicalImage&) override {}
  void destroy_buffer(fwrk::PhysicalBuffer&) override {};
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
  void jfa_pass(vk::CommandBuffer cmd);
  void sdf_pass(vk::CommandBuffer cmd);
  void cascades_pass(vk::CommandBuffer cmd);
  void merge_pass(vk::CommandBuffer cmd);
  void light_pass(vk::CommandBuffer cmd);
  void pull_push_pass(vk::CommandBuffer cmd);
  void composite_pass(vk::CommandBuffer cmd);
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
  EmptyAllocator empty_allocator_;
  fwrk::Context context_;

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

  vk::UniquePipelineLayout light_pipeline_layout_;
  vk::UniqueShaderModule light_shader_module_;
  vk::UniquePipeline light_pipeline_;

  vk::UniquePipelineLayout pull_push_pipeline_layout_;
  vk::UniqueShaderModule pull_push_shader_module_;
  vk::UniquePipeline pull_pipeline_;
  vk::UniquePipeline push_pipeline_;

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
  vk::UniqueImageView scene_view_;
  fwrk::ResourceID scene_id_;

  std::optional<Image> jfa_image_;
  vk::UniqueImageView jfa_view_;
  fwrk::ResourceID jfa_id_;

  std::optional<Image> sdf_image_;
  vk::UniqueImageView sdf_view_;
  fwrk::ResourceID sdf_id_;

  std::optional<Image> cascades_image_;
  vk::UniqueImageView cascades_view_;
  fwrk::ResourceID cascades_id_;

  std::optional<Image> light_image_;
  std::vector<vk::UniqueImageView> light_views_;
  fwrk::ResourceID light_id_;

  std::optional<Image> composite_image_;
  vk::UniqueImageView composite_view_;
  fwrk::ResourceID composite_id_;

  std::optional<Buffer> material_buffer_;
  std::optional<Buffer> ubo_;

  std::vector<fwrk::ResourceID> swapchain_imports_;
  fwrk::ResourceID swapchain_proxy_;

  uint32_t current_frame_{};
  uint32_t image_index_{};

  bool should_compile_ = true;

  std::vector<Material> materials_;
  uint8_t draw_material_ = 1;
  bool should_draw = false;
  std::pair<uint32_t, uint32_t> draw_pos_;

  uint32_t final_jfa_{0};

  uint32_t cascades_width_{0};
  uint32_t cascades_height_{0};
  uint32_t base_probe_size_{0};
  float base_probe_spacing_{0.0f};
  float base_probe_length_{0.0f};

  uint32_t light_mips_{0};

  void import_swapchain_images();
  void import_images();

  void create_images();
  void create_cascade_image();

  void update_descriptor();

  static VkSurfaceKHR create_surface(const Window& window, const Instance& instance);
};
