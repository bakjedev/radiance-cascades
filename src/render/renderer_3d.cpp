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

void Renderer3D::upload_vertices(const std::vector<Vertex>& vertices, const std::vector<uint32_t>& indices)
{
  device_.get().waitIdle(); // bit blunt, could queue these uploads instead and handle the sync in next frame.

  BufferDesc staging_desc{
      .usage = vk::BufferUsageFlagBits::eTransferSrc,
      .memory_usage = VMA_MEMORY_USAGE_AUTO,
      .alloc_flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT};

  const auto vertex_size = vertices.size() * sizeof(Vertex);
  staging_desc.set_size(vertex_size);
  const Buffer staging_vertex{device_.get_allocator(), device_.get(), staging_desc};
  std::memcpy(staging_vertex.get_mapped_data(), vertices.data(), vertex_size);

  const auto index_size = indices.size() * sizeof(uint32_t);
  staging_desc.set_size(index_size);
  const Buffer staging_index{device_.get_allocator(), device_.get(), staging_desc};
  std::memcpy(staging_index.get_mapped_data(), indices.data(), index_size);

  vertex_buffer_.emplace(
      device_.get_allocator(), device_.get(),
      BufferDesc{.size = vertex_size,
                 .usage = vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eVertexBuffer,
                 .memory_usage = VMA_MEMORY_USAGE_AUTO,
                 .debug_name = "Vertex Buffer"});

  index_buffer_.emplace(
      device_.get_allocator(), device_.get(),
      BufferDesc{.size = index_size,
                 .usage = vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eIndexBuffer,
                 .memory_usage = VMA_MEMORY_USAGE_AUTO,
                 .debug_name = "Index Buffer"});

  immediate_.submit([&](vk::CommandBuffer cmd) {
    vk::BufferCopy2 region{};
    region.setSize(vertex_size);

    vk::CopyBufferInfo2 info{};
    info.setSrcBuffer(staging_vertex.buffer());
    info.setDstBuffer(vertex_buffer_->buffer());
    info.setRegions(region);
    cmd.copyBuffer2(&info);

    region.setSize(index_size);
    info.setSrcBuffer(staging_index.buffer());
    info.setDstBuffer(index_buffer_->buffer());
    cmd.copyBuffer2(&info);
  });
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
