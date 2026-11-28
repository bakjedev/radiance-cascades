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
  GPUPrimitive create_primitive(std::span<const Vertex> vertices, std::span<const uint32_t> indices,
                                const glm::vec4& bounds);

  void submit_geometry(const GeometrySubmission& submission);
  void submit_light(const PointLightSubmission& submission);
  void submit_light(const SpotLightSubmission& submission);

private:
  static constexpr uint32_t max_geometries = 1 << 16;
  static constexpr uint32_t filter_count = 5;

  static constexpr uint32_t max_point_lights = 512;
  static constexpr uint32_t max_spot_lights = 512;
  static constexpr uint32_t max_lights = max_point_lights + max_spot_lights;

  static constexpr uint32_t max_point_shadows = 4;
  static constexpr uint32_t max_spot_shadows = 4;

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
    uint32_t alpha_mode;
    float alpha_cutoff;
    uint32_t pad_0;
    uint32_t pad_1;
  };

  struct Primitive {
    uint32_t first_index;
    uint32_t index_count;
    uint32_t vertex_offset;
    uint32_t pad{};
    glm::vec4 local_bounds;
  };

  struct Geometry {
    glm::mat4 transform;
    glm::vec4 bounds;
    GPUPrimitive primitive;
    GPUMaterial material;
    uint32_t flags;
    uint32_t pad{};
  };

  struct CullFilter {
    std::array<glm::vec4, 6> planes;
    uint32_t include_mask;
    uint32_t exclude_mask;
    uint32_t cull_flags;
    uint32_t base;
  };

  struct Light {
    glm::vec3 position;
    float range;
    glm::vec3 direction;
    float angle_size;
    glm::vec3 radiance;
    float angle_offset;
  };

  struct FrameData {
    glm::mat4 inv_view_proj;
    alignas(16) glm::vec3 camera_pos;
    alignas(16) glm::vec3 sun_dir;
    alignas(16) glm::vec3 sun_color;
    alignas(16) glm::vec3 ambient;
    alignas(16) glm::vec3 ambient_indirect;
  };

  struct CascadedShadows {
    static constexpr auto Resolution = 2048;
    static constexpr auto Count = 4;
    static constexpr auto Distance = 50.0f;
    static constexpr auto Lambda = 0.75f;
    static constexpr float FilterRadius = 0.015f;
    static constexpr float Blend = 0.1f;
    glm::mat4 world_to_shadow_clip[Count];
    glm::vec4 world_units_per_texel;
    glm::vec4 filter_radius_per_texel;
    float inv_res;
    float blend;
    glm::vec2 pad;
  };

  struct LightsData {
    std::array<glm::mat4, max_point_shadows> point_view_proj;
    std::array<glm::mat4, max_spot_shadows> spot_view_proj;
    uint32_t point_lights_base;
    uint32_t spot_lights_base;
    uint32_t point_light_count;
    uint32_t spot_light_count;
  };

  RenderInfo info_{};

  std::vector<vk::UniqueSampler> samplers_;
  std::vector<Material> materials_;
  bool upload_materials_ = false;

  std::vector<GeometrySubmission> geometry_submissions_;
  uint32_t geometries_count_{};

  std::vector<Vertex> vertices_;
  std::vector<uint32_t> indices_;
  std::vector<Primitive> primitives_;
  std::optional<Buffer> vertex_buffer_;
  std::optional<Buffer> index_buffer_;
  std::optional<Buffer> primitive_buffer_;
  bool upload_primitives_ = false;

  std::optional<Buffer> material_buffer_;

  std::optional<Buffer> indirect_buffer_;
  fwrk::ResourceID indirect_id;
  std::optional<Buffer> geometries_buffer_;

  std::optional<Buffer> filter_buffer_;

  std::optional<Buffer> count_buffer;
  fwrk::ResourceID count_id;

  std::vector<Light> point_lights_;
  std::vector<Light> spot_lights_;
  std::optional<Buffer> lights_buffer_;

  std::optional<Buffer> ubo_buffer_;
  FrameData frame_data_{};
  CascadedShadows cascades_{};
  LightsData lights_data_{};
  vk::DeviceSize frame_data_ubo_size_{};
  vk::DeviceSize cascaded_shadows_ubo_size_{};
  vk::DeviceSize lights_data_ubo_size_;
  vk::DeviceSize cascades_ubo_base_{};
  vk::DeviceSize lights_data_ubo_base_;

  vk::UniqueSampler cascaded_shadow_sampler_;

  vk::UniquePipelineLayout cull_pipeline_layout_;
  vk::UniqueShaderModule cull_shader_module_;
  vk::UniquePipeline cull_pipeline_;

  vk::UniquePipelineLayout raster_pipeline_layout_;
  vk::UniqueShaderModule raster_shader_module_;
  vk::UniquePipeline raster_pipeline_;

  vk::UniquePipelineLayout shadow_pipeline_layout_;
  vk::UniqueShaderModule shadow_shader_module_;
  vk::UniquePipeline shadow_pipeline_;

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

  std::optional<Image> cascaded_shadow_image_;
  vk::UniqueImageView cascaded_shadow_view_;
  fwrk::ResourceID cascaded_shadow_id_;

  std::optional<Image> composite_image_;
  vk::UniqueImageView composite_view_;
  fwrk::ResourceID composite_id_;

  void compile();

  void imgui_pass(vk::CommandBuffer cmd) const;
  void clear_count_pass(vk::CommandBuffer cmd) const;
  void cull_pass(vk::CommandBuffer cmd) const;
  void raster_pass(vk::CommandBuffer cmd) const;
  void shadow_pass(vk::CommandBuffer cmd, uint32_t cascade) const;
  void composite_pass(vk::CommandBuffer cmd) const;
  void blit_pass(vk::CommandBuffer cmd) const;

  void upload_primitives();

  void upload_geometry();

  void upload_lights();

  void upload_materials();

  void update_filters() const;

  void update_frame_data();

  void create_render_targets(vk::Extent2D extent);

  void update_descriptors();

  void update_cascades();

  static uint32_t pack_texture_index(const MaterialInfo::Textures::Texture& texture);

  static std::array<glm::vec4, 6> extract_frustum_planes(const glm::mat4& matrix);
};

static_assert(RendererInterface<Renderer3D>);
