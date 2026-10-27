#pragma once
#include "backend/buffer.hpp"
#include "backend/device.hpp"
#include "backend/frame.hpp"
#include "backend/image.hpp"
#include "backend/immediate_submit.hpp"
#include "backend/instance.hpp"
#include "backend/swapchain.hpp"
#include "framework/context.hpp"

struct ImDrawData;

class Renderer2D {
public:
  struct RendererCreateInfo {
    std::span<const char* const> instance_extensions;
    std::function<VkSurfaceKHR(VkInstance)> create_surface;
    VkExtent2D initial_extent;
    std::function<std::vector<std::byte>(const std::string&)> load_shader;
  };

  struct RenderInfo {
    std::pair<int, int> window_size;
    std::pair<int, int> mouse_pos;
    ImDrawData* imgui_draw_data;
  };

  struct Renderer2DConfig {
    struct DrawConfig {
      uint32_t size{50};
      uint8_t material{1};
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
      uint32_t bounce{1};
    } cascades;

    struct CompositeConfig {
      uint32_t blur_radius{120};
      float rim{12.0f};
    } composite;
  };

  struct Material {
    float color[3];
    float padding{};
    float radiance[3];
    float paddington{};
  };

  struct Texture {
    uint32_t id;
  };

  struct Sprite {
    Texture texture;
    std::pair<float, float> pos;
    std::pair<float, float> size = {1.0f, 1.0f};
    std::tuple<float, float, float> tint = {1.0f, 1.0f, 1.0f};
  };

  explicit Renderer2D(const RendererCreateInfo& create_info);
  ~Renderer2D();

  void render(RenderInfo info);

  void window_resize(int width, int height);

  Texture create_texture(std::span<const uint8_t> data, const std::pair<uint32_t, uint32_t>& size);

  void plot();
  void draw_sprite(Sprite sprite);

  static void imgui_new_frame();
  void request_recompile();

  [[nodiscard]] fwrk::Graph::FrameTimings get_timings();

  Renderer2DConfig& config() { return config_; }
  const std::vector<Material>& materials() { return materials_; }

private:
  static constexpr uint32_t frames_in_flight = 2;
  static constexpr uint32_t max_sprites = 4096;

  struct SpriteGPU {
    float tint[3];
    uint32_t texture;
    float pos[2];
    float size[2];

    explicit SpriteGPU(const Sprite& sprite) :
        tint(std::get<0>(sprite.tint), std::get<1>(sprite.tint), std::get<2>(sprite.tint)), texture(sprite.texture.id),
        pos(sprite.pos.first, sprite.pos.second), size(sprite.size.first, sprite.size.second)
    {
    }
  };

  bool begin_frame();
  void run_frame();
  void end_frame();

  void compile();

  void draw_pass(vk::CommandBuffer cmd);
  void convert_pass(vk::CommandBuffer cmd);
  void sprite_pass(vk::CommandBuffer cmd);
  void jfa_pass(vk::CommandBuffer cmd);
  void sdf_pass(vk::CommandBuffer cmd);
  void cascades_pass(vk::CommandBuffer cmd, bool first = true);
  void light_pass(vk::CommandBuffer cmd);
  void blur_pass(vk::CommandBuffer cmd);
  void composite_pass(vk::CommandBuffer cmd);
  void blit_pass(vk::CommandBuffer cmd);
  void imgui_pass(vk::CommandBuffer cmd) const;

  Renderer2DConfig config_;
  RenderInfo info_;

  Instance instance_;
  Device device_;
  Swapchain swapchain_;
  std::array<Frame, frames_in_flight> frames_;
  std::vector<vk::UniqueSemaphore> submit_semaphores_;
  fwrk::Context context_;
  ImmediateSubmit immediate_;

  vk::UniquePipelineLayout draw_pipeline_layout_;
  vk::UniqueShaderModule draw_shader_module_;
  vk::UniquePipeline draw_pipeline_;

  vk::UniquePipelineLayout convert_pipeline_layout_;
  vk::UniqueShaderModule convert_shader_module_;
  vk::UniquePipeline convert_pipeline_;

  vk::UniquePipelineLayout sprite_pipeline_layout_;
  vk::UniqueShaderModule sprite_shader_module_;
  vk::UniquePipeline sprite_pipeline_;

  vk::UniquePipelineLayout jfa_pipeline_layout_;
  vk::UniqueShaderModule jfa_shader_module_;
  vk::UniquePipeline jfa_pipeline_;

  vk::UniquePipelineLayout sdf_pipeline_layout_;
  vk::UniqueShaderModule sdf_shader_module_;
  vk::UniquePipeline sdf_pipeline_;

  vk::UniquePipelineLayout cascades_pipeline_layout_;
  vk::UniqueShaderModule cascades_shader_module_;
  vk::UniquePipeline cascades_pipeline_;

  vk::UniquePipelineLayout light_pipeline_layout_;
  vk::UniqueShaderModule light_shader_module_;
  vk::UniquePipeline light_pipeline_;

  vk::UniquePipelineLayout blur_pipeline_layout_;
  vk::UniqueShaderModule blur_shader_module_;
  vk::UniquePipeline blur_pipeline_;

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

  std::optional<Image> draw_image_;
  vk::UniqueImageView draw_view_;
  fwrk::ResourceID draw_id_;

  std::optional<Image> scene_color_image_;
  vk::UniqueImageView scene_color_view_;
  fwrk::ResourceID scene_color_id_;

  std::optional<Image> scene_radiance_image_;
  vk::UniqueImageView scene_radiance_view_;
  fwrk::ResourceID scene_radiance_id_;

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
  vk::UniqueImageView light_view_;
  fwrk::ResourceID light_id_;

  std::optional<Image> blur_image_;
  vk::UniqueImageView blur_view_;
  fwrk::ResourceID blur_id_;

  std::optional<Image> composite_image_;
  vk::UniqueImageView composite_view_;
  fwrk::ResourceID composite_id_;

  std::vector<std::pair<Image, vk::UniqueImageView>> texture_images_;

  std::optional<Buffer> material_buffer_;
  std::optional<Buffer> ubo_;

  std::vector<fwrk::ResourceID> swapchain_imports_;
  fwrk::ResourceID swapchain_proxy_;

  uint32_t current_frame_{};
  uint32_t image_index_{};

  bool should_compile_ = true;

  std::vector<Material> materials_;
  bool should_draw_ = false;
  bool did_draw_ = false;

  std::vector<SpriteGPU> sprites_;
  void* sprites_mapped = nullptr;

  uint32_t final_jfa_{0};
  uint32_t cascades_width_{0};
  uint32_t cascades_height_{0};
  uint32_t base_probe_size_{0};
  float base_probe_spacing_{0.0f};
  float base_probe_length_{0.0f};

  void import_swapchain_images();
  void import_images();

  void create_images();
  void create_cascade_image();

  void update_descriptor();

  void upload_sprites() const;

  [[nodiscard]] std::pair<int32_t, int32_t> screen_to_pos(const std::pair<int32_t, int32_t>& screen) const;

  [[nodiscard]] fwrk::Context create_fwrk_context() const;
};
