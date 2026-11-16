#include "renderer_3d.hpp"

#include "imgui_impl_vulkan.h"

Renderer3D::Renderer3D(const RendererCreateInfo& create_info) : RenderBackend(create_info) {}

Renderer3D::~Renderer3D() { device_.get().waitIdle(); }

void Renderer3D::render(RenderInfo info)
{
  info_ = std::move(info);
  if (!begin_frame({info_.window_size.first, info_.window_size.second})) {
    return;
  }

  if (should_compile_) compile();

  execute_frame();
  end_frame();
}

GPUSampler Renderer3D::create_sampler(const Filter min_filter, const Filter mag_filter, const Wrap wrap_s,
                                      const Wrap wrap_t)
{
  const auto id = static_cast<uint32_t>(samplers_.size());
  auto& sampler = samplers_.emplace_back();

  // little bit of duplicated code incoming

  vk::SamplerCreateInfo create_info{};
  switch (min_filter) {
    case Filter::Nearest:
      create_info.setMinFilter(vk::Filter::eNearest);
      break;
    case Filter::Linear:
      create_info.setMinFilter(vk::Filter::eLinear);
      break;
    case Filter::NearestMipMapNearest:
      create_info.setMinFilter(vk::Filter::eNearest);
      create_info.setMipmapMode(vk::SamplerMipmapMode::eNearest);
      break;
    case Filter::LinearMipMapNearest:
      create_info.setMinFilter(vk::Filter::eLinear);
      create_info.setMipmapMode(vk::SamplerMipmapMode::eNearest);
      break;
    case Filter::NearestMipMapLinear:
      create_info.setMinFilter(vk::Filter::eNearest);
      create_info.setMipmapMode(vk::SamplerMipmapMode::eLinear);
      break;
    case Filter::LinearMipMapLinear:
      create_info.setMinFilter(vk::Filter::eLinear);
      create_info.setMipmapMode(vk::SamplerMipmapMode::eLinear);
      break;
  }

  switch (mag_filter) {
    case Filter::Nearest:
    case Filter::NearestMipMapNearest:
    case Filter::NearestMipMapLinear:
      create_info.setMagFilter(vk::Filter::eNearest);
      break;
    case Filter::Linear:
    case Filter::LinearMipMapNearest:
    case Filter::LinearMipMapLinear:
      create_info.setMagFilter(vk::Filter::eLinear);
      break;
  }

  switch (wrap_s) {
    case Wrap::ClampToEdge:
      create_info.setAddressModeU(vk::SamplerAddressMode::eClampToEdge);
      break;
    case Wrap::MirroredRepeat:
      create_info.setAddressModeU(vk::SamplerAddressMode::eMirroredRepeat);
      break;
    case Wrap::Repeat:
      create_info.setAddressModeU(vk::SamplerAddressMode::eRepeat);
      break;
  }

  switch (wrap_t) {
    case Wrap::ClampToEdge:
      create_info.setAddressModeV(vk::SamplerAddressMode::eClampToEdge);
      break;
    case Wrap::MirroredRepeat:
      create_info.setAddressModeV(vk::SamplerAddressMode::eMirroredRepeat);
      break;
    case Wrap::Repeat:
      create_info.setAddressModeV(vk::SamplerAddressMode::eRepeat);
      break;
  }

  sampler = device_.get().createSamplerUnique(create_info);
  return {id};
}

GPUMaterial Renderer3D::create_material(const MaterialInfo& material)
{
  const auto id = static_cast<uint32_t>(materials_.size());
  materials_.emplace_back(material.base_color, material.emissive_color, material.metallic, material.roughness,
                          material.normal_scale, material.occlusion_scale, pack_texture_index(material.textures.albedo),
                          pack_texture_index(material.textures.metal_roughness),
                          pack_texture_index(material.textures.normal), pack_texture_index(material.textures.occlusion),
                          pack_texture_index(material.textures.emissive));
  return {id};
}

void Renderer3D::upload_vertices([[maybe_unused]] const std::vector<Vertex>& vertices,
                                 [[maybe_unused]] const std::vector<uint32_t>& indices)
{
  // WIP
}

void Renderer3D::compile()
{
  fwrk::Graph& graph = context_.graph();

  graph.add_graphics_pass("ImGui")
      .set_color_attachment({
          .resource = {.id = swapchain_proxy_},
          .load_op = fwrk::LoadOp::Clear,
          .store_op = fwrk::StoreOp::Store,
      })
      .set_execute([this](vk::CommandBuffer cmd) { imgui_pass(cmd); });

  graph.set_image_end_state(swapchain_proxy_,
                            {VK_ACCESS_2_NONE, VK_PIPELINE_STAGE_2_NONE, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR});

  graph.compile();
}

void Renderer3D::imgui_pass(vk::CommandBuffer cmd) const
{
  ImGui_ImplVulkan_RenderDrawData(info_.imgui_draw_data, cmd);
}

uint32_t Renderer3D::pack_texture_index(const MaterialInfo::Textures::Texture& texture)
{
  assert((texture.index < MaterialTextureMask || texture.index == UINT32_MAX) && texture.tex_coord < 256);
  return pack_material_texture(texture.index == UINT32_MAX ? MaterialTextureMask : texture.index, texture.tex_coord);
}
