#pragma once
#include "render_backend.hpp"
#include "renderer_interface.hpp"

class Renderer3D : public RenderBackend {
public:
  explicit Renderer3D(const RendererCreateInfo& create_info);
  ~Renderer3D();
  void render(RenderInfo info);

  GPUSampler create_sampler(Filter min_filter, Filter mag_filter, Wrap wrap_s, Wrap wrap_t);
  GPUMaterial create_material(const MaterialInfo& material);

private:
  static constexpr uint32_t MaterialTextureBits = 24;
  static constexpr uint32_t MaterialTextureMask = (1u << MaterialTextureBits) - 1;
  static constexpr uint32_t pack_material_texture(const uint32_t index, const uint32_t tex_coord)
  {
    return (index & MaterialTextureMask) | (tex_coord << MaterialTextureBits);
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

  RenderInfo info_{};

  std::vector<vk::UniqueSampler> samplers_;
  std::vector<Material> materials_;

  void compile();

  void imgui_pass(vk::CommandBuffer cmd) const;

  static uint32_t pack_texture_index(const MaterialInfo::Textures::Texture& texture);
};

static_assert(RendererInterface<Renderer3D>);
