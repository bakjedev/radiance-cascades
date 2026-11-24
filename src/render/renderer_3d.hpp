#pragma once
#include "render_backend.hpp"
#include "renderer_interface.hpp"
#include "src/vulkan/buffer.hpp"

class Renderer3D : public RenderBackend {
public:
  explicit Renderer3D(const RendererCreateInfo& create_info);
  ~Renderer3D();
  void render(RenderInfo info);

  GPUSampler create_sampler(Filter min_filter, Filter mag_filter, Wrap wrap_s, Wrap wrap_t);
  GPUMaterial create_material(const MaterialInfo& material);
  void upload_vertices(const std::vector<Vertex>& vertices, const std::vector<uint32_t>& indices);

  void draw_geometry(const GeometrySubmission& submission);

private:
  static constexpr uint32_t max_geometries = 1 << 16;

  static constexpr uint32_t MaterialTextureBits = 23;
  static constexpr uint32_t MaterialTextureMask = (1u << MaterialTextureBits) - 1;
  static constexpr uint32_t MaterialSamplerShift = MaterialTextureBits + 1;
  static constexpr uint32_t MaterialSamplerMask = 0xFFu;

  static constexpr uint32_t pack_material_texture(const uint32_t index, const uint32_t tex_coord,
                                                  const uint32_t sampler)
  {
    return (index & MaterialTextureMask) | ((tex_coord & 1u) << MaterialTextureBits) |
           ((sampler & MaterialSamplerMask) << MaterialSamplerShift);
  }

  struct Material {
    glm::vec4 base_color;
    glm::vec3 emissive_color;
    float metallic;
    float roughness;
    float normal_scale;
    float occlusion_scale;
    uint32_t albedo_tex;
    uint32_t metal_roughness_tex;
    uint32_t normal_tex;
    uint32_t occlusion_tex;
    uint32_t emissive_tex;
  };

  struct Geometry {
    glm::mat4 transform;
    GPUMaterial material;
  };

  struct FrameData {
    glm::mat4 inv_view_proj;
    alignas(16) glm::vec3 camera_pos;
    alignas(16) glm::vec3 sun_dir;
    alignas(16) glm::vec3 sun_color;
    alignas(16) glm::vec3 ambient;
    alignas(16) glm::vec3 ambient_indirect;
  };

  RenderInfo info_{};

  std::vector<vk::UniqueSampler> samplers_;
  std::vector<Material> materials_;
  bool upload_materials_ = false;

  std::vector<GeometrySubmission> geometry_submissions_;

  std::optional<Buffer> vertex_buffer_;
  std::optional<Buffer> index_buffer_;

  std::optional<Buffer> material_buffer_;

  std::optional<Buffer> indirect_buffer_;
  std::optional<Buffer> geometries_buffer_;

  std::optional<Buffer> frame_data_buffer_;
  FrameData frame_data_{};

  vk::UniquePipelineLayout raster_pipeline_layout_;
  vk::UniqueShaderModule raster_shader_module_;
  vk::UniquePipeline raster_pipeline_;

  vk::UniquePipelineLayout composite_pipeline_layout_;
  vk::UniqueShaderModule composite_shader_module_;
  vk::UniquePipeline composite_pipeline_;

  vk::UniquePipelineLayout blit_pipeline_layout_;
  vk::UniqueShaderModule blit_shader_module_;
  vk::UniquePipeline blit_pipeline_;

  std::optional<Image> albedo_image_;
  std::optional<Image> normal_image_;
  std::optional<Image> pbr_image_;
  std::optional<Image> emissive_image_;
  std::optional<Image> depth_image_;

  vk::UniqueImageView albedo_view_;
  vk::UniqueImageView normal_view_;
  vk::UniqueImageView pbr_view_;
  vk::UniqueImageView emissive_view_;
  vk::UniqueImageView depth_view_;

  fwrk::ResourceID albedo_id_;
  fwrk::ResourceID normal_id_;
  fwrk::ResourceID pbr_id_;
  fwrk::ResourceID emissive_id_;
  fwrk::ResourceID depth_id_;

  std::optional<Image> composite_image_;
  vk::UniqueImageView composite_view_;
  fwrk::ResourceID composite_id_;

  uint32_t geometry_count_ = 0;

  void compile();

  void imgui_pass(vk::CommandBuffer cmd) const;
  void raster_pass(vk::CommandBuffer cmd) const;
  void composite_pass(vk::CommandBuffer cmd) const;
  void blit_pass(vk::CommandBuffer cmd) const;

  void upload_geometry();

  void upload_materials();

  void update_frame_data();

  void create_render_targets(vk::Extent2D extent);

  void update_descriptors();

  static uint32_t pack_texture_index(const MaterialInfo::Textures::Texture& texture);
};

static_assert(RendererInterface<Renderer3D>);
