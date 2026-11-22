#include "renderer_3d.hpp"

#include "imgui_impl_vulkan.h"
#include "src/vulkan/descriptor.hpp"
#include "src/vulkan/pipeline.hpp"

namespace {
  struct RasterPushConstant {
    glm::mat4 view_proj;
    uint32_t geometries_id;
    uint32_t materials_id;
    uint32_t textures_id;
    uint32_t samplers_id;
  };

  enum TextureDescriptors {
    Textures = 0,
  };

  enum BufferDescriptors {
    GeometriesBuffer = 0,
    MaterialBuffer,
  };

  enum SamplerDescriptors {
    TextureSamplers,
  };
} // namespace

Renderer3D::Renderer3D(const RendererCreateInfo& create_info) : RenderBackend(create_info)
{
  indirect_buffer_.emplace(
      device_.get_allocator(), device_.get(),
      BufferDesc{
          .size = sizeof(vk::DrawIndexedIndirectCommand) * max_geometries * frames_in_flight,
          .usage = vk::BufferUsageFlagBits::eIndirectBuffer,
          .memory_usage = VMA_MEMORY_USAGE_AUTO,
          .alloc_flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT,
          .debug_name = "Indirect buffer"});

  geometries_buffer_.emplace(device_.get_allocator(), device_.get(),
                             BufferDesc{.size = sizeof(Geometry) * max_geometries * frames_in_flight,
                                        .usage = vk::BufferUsageFlagBits::eStorageBuffer,
                                        .memory_usage = VMA_MEMORY_USAGE_AUTO,
                                        .alloc_flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                                                       VMA_ALLOCATION_CREATE_MAPPED_BIT,
                                        .debug_name = "Geometries buffer"});

  std::vector<vk::DebugUtilsObjectNameInfoEXT> debug_object_names;
  debug_object_names.reserve(1);

  auto handle_to_uint64_t = []<typename T>(T handle) -> uint64_t {
    return reinterpret_cast<uint64_t>(static_cast<T::CType>(handle));
  };

  {
    vk::PushConstantRange raster_push{vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0,
                                      sizeof(RasterPushConstant)};

    const auto raster_shader = create_info.load_shader("raster.slang.spv");
    raster_shader_module_ = create_shader_module(device_.get(), raster_shader);

    raster_pipeline_layout_ =
        create_pipeline_layout(device_.get(), {&bindless_descriptor_set_layout_.get(), 1}, {&raster_push, 1});

    GraphicsPipelineDesc raster_pipeline_desc{};
    raster_pipeline_desc.stages.emplace_back(vk::ShaderStageFlagBits::eVertex, raster_shader_module_.get(), "vs_main");
    raster_pipeline_desc.stages.emplace_back(vk::ShaderStageFlagBits::eFragment, raster_shader_module_.get(),
                                             "fs_main");

    raster_pipeline_desc.vertex_bindings = {
        {0, sizeof(Vertex), vk::VertexInputRate::eVertex},
    };

    raster_pipeline_desc.vertex_attributes = {
        {0, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, position)},
        {1, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, normal)},
        {2, 0, vk::Format::eR32G32Sfloat, offsetof(Vertex, tex_coord_0)},
        {3, 0, vk::Format::eR32G32Sfloat, offsetof(Vertex, tex_coord_1)},
        {4, 0, vk::Format::eR32G32B32A32Sfloat, offsetof(Vertex, tangent)},
    };

    raster_pipeline_desc.add_attachment(vk::Format::eR8G8B8A8Srgb);
    raster_pipeline_desc.add_attachment(vk::Format::eR16G16Snorm);
    raster_pipeline_desc.add_attachment(vk::Format::eR8G8B8A8Unorm);
    raster_pipeline_desc.add_attachment(vk::Format::eB10G11R11UfloatPack32);

    raster_pipeline_desc.depth_format = vk::Format::eD32Sfloat;

    raster_pipeline_desc.depth_stencil.depthTestEnable = vk::True;
    raster_pipeline_desc.depth_stencil.depthWriteEnable = vk::True;
    raster_pipeline_desc.depth_stencil.depthCompareOp = vk::CompareOp::eGreater;

    raster_pipeline_desc.layout = raster_pipeline_layout_.get();

    raster_pipeline_ = create_graphics_pipeline(device_.get(), raster_pipeline_desc);

    debug_object_names.emplace_back(vk::ObjectType::ePipeline, handle_to_uint64_t(raster_pipeline_.get()),
                                    "Sprite Pipeline");
  }

  for (auto& info: debug_object_names) {
    if (device_.get().setDebugUtilsObjectNameEXT(&info) != vk::Result::eSuccess) {
      throw std::runtime_error("Failed to set object name on pipeline");
    }
  }

  create_render_targets(create_info.initial_extent);

  DescriptorWriter{}
      .add_buffer(1, GeometriesBuffer, vk::DescriptorType::eStorageBuffer, geometries_buffer_->buffer())
      .update(device_.get(), bindless_descriptor_set_);
}

Renderer3D::~Renderer3D() { device_.get().waitIdle(); }

void Renderer3D::render(RenderInfo info)
{
  info_ = std::move(info);

  const vk::Extent2D extent = {info_.window_size.first, info_.window_size.second};
  if (needs_recreate_ || extent != swapchain_.extent()) create_render_targets(extent);

  if (!begin_frame(extent)) {
    return;
  }

  if (upload_materials_) {
    upload_materials();
    upload_materials_ = false;
  }

  upload_geometry();

  if (should_compile_) {
    should_compile_ = false;
    compile();
  }

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
  upload_materials_ = true;
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

void Renderer3D::draw_geometry(const GeometrySubmission& submission) { geometry_submissions_.push_back(submission); }

void Renderer3D::compile()
{
  fwrk::Graph& graph = context_.graph();

  update_descriptors();

  graph.add_graphics_pass("Raster")
      .set_color_attachment(
          {.resource = {.id = albedo_id_}, .load_op = fwrk::LoadOp::Clear, .store_op = fwrk::StoreOp::Store})
      .set_color_attachment(
          {.resource = {.id = normal_id_}, .load_op = fwrk::LoadOp::Clear, .store_op = fwrk::StoreOp::Store})
      .set_color_attachment(
          {.resource = {.id = pbr_id_}, .load_op = fwrk::LoadOp::Clear, .store_op = fwrk::StoreOp::Store})
      .set_color_attachment(
          {.resource = {.id = emissive_id_}, .load_op = fwrk::LoadOp::Clear, .store_op = fwrk::StoreOp::Store})
      .set_depth_attachment(
          {.resource = {.id = depth_id_}, .load_op = fwrk::LoadOp::Clear, .store_op = fwrk::StoreOp::Store})
      .set_execute([this](vk::CommandBuffer cmd) { raster_pass(cmd); });

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

void Renderer3D::raster_pass(vk::CommandBuffer cmd) const
{
  cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, raster_pipeline_.get());
  cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, raster_pipeline_layout_.get(), 0, 1,
                         &bindless_descriptor_set_, 0, nullptr);

  const vk::Viewport viewport{
      0.0f, 0.0f, static_cast<float>(info_.window_size.first), static_cast<float>(info_.window_size.second),
      0.0f, 1.0f};
  cmd.setViewport(0, 1, &viewport);

  const vk::Rect2D scissor{vk::Offset2D{0, 0}, vk::Extent2D{info_.window_size.first, info_.window_size.second}};
  cmd.setScissor(0, 1, &scissor);

  const RasterPushConstant push_constant{.view_proj = info_.projection * info_.view,
                                         .geometries_id = GeometriesBuffer,
                                         .materials_id = MaterialBuffer,
                                         .textures_id = Textures,
                                         .samplers_id = TextureSamplers};

  cmd.pushConstants(raster_pipeline_layout_.get(),
                    vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0,
                    sizeof(RasterPushConstant), &push_constant);

  constexpr VkDeviceSize offset = 0;
  const auto vertex_buffer = vertex_buffer_->buffer();
  const auto index_buffer = index_buffer_->buffer();
  cmd.bindVertexBuffers(0, 1, &vertex_buffer, &offset);
  cmd.bindIndexBuffer(index_buffer, offset, vk::IndexType::eUint32);

  constexpr auto stride = sizeof(vk::DrawIndexedIndirectCommand);
  cmd.drawIndexedIndirect(indirect_buffer_->buffer(), stride * max_geometries * current_frame_, geometry_count_,
                          stride);
}

void Renderer3D::upload_geometry()
{
  constexpr auto indirect_region = sizeof(vk::DrawIndexedIndirectCommand) * max_geometries;
  constexpr auto geometries_region = sizeof(vk::DrawIndexedIndirectCommand) * max_geometries;

  auto* indirect_mapping = reinterpret_cast<vk::DrawIndexedIndirectCommand*>(
      indirect_buffer_->get_mapped_data<std::byte>() + indirect_region * current_frame_);
  auto* geometries_mapping = reinterpret_cast<Geometry*>(geometries_buffer_->get_mapped_data<std::byte>() +
                                                         geometries_region * current_frame_);

  geometry_count_ = static_cast<uint32_t>(geometry_submissions_.size());
  for (uint32_t i = 0; i < geometry_count_; i++) {
    const auto& [range, material, transform] = geometry_submissions_[i];
    indirect_mapping[i] = vk::DrawIndexedIndirectCommand{}
                              .setIndexCount(range.index_count)
                              .setInstanceCount(1)
                              .setFirstIndex(range.first_index)
                              .setVertexOffset(range.vertex_offset)
                              .setFirstInstance(i);
    geometries_mapping[i] = Geometry{.transform = transform, .material = material};
  }
  geometry_submissions_.clear();
}

void Renderer3D::upload_materials()
{
  device_.get().waitIdle(); // bit blunt, could queue these uploads instead and handle the sync in next frame.

  const auto material_size = materials_.size() * sizeof(Material);
  const Buffer staging{device_.get_allocator(), device_.get(),
                       BufferDesc{.size = material_size,
                                  .usage = vk::BufferUsageFlagBits::eTransferSrc,
                                  .memory_usage = VMA_MEMORY_USAGE_AUTO,
                                  .alloc_flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                                                 VMA_ALLOCATION_CREATE_MAPPED_BIT}};
  std::memcpy(staging.get_mapped_data(), materials_.data(), material_size);

  material_buffer_.emplace(
      device_.get_allocator(), device_.get(),
      BufferDesc{.size = material_size,
                 .usage = vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eStorageBuffer,
                 .memory_usage = VMA_MEMORY_USAGE_AUTO,
                 .debug_name = "Material buffer"});

  immediate_.submit([&](vk::CommandBuffer cmd) {
    vk::BufferCopy2 region{};
    region.setSize(material_size);

    vk::CopyBufferInfo2 info{};
    info.setSrcBuffer(staging.buffer());
    info.setDstBuffer(material_buffer_->buffer());
    info.setRegions(region);
    cmd.copyBuffer2(&info);
  });

  DescriptorWriter{}
      .add_buffer(1, MaterialBuffer, vk::DescriptorType::eStorageBuffer, material_buffer_->buffer())
      .update(device_.get(), bindless_descriptor_set_);
}

void Renderer3D::create_render_targets(const vk::Extent2D extent)
{
  albedo_image_.emplace(device_.get_allocator(), device_.get(),
                        ImageDesc{.format = vk::Format::eR8G8B8A8Srgb,
                                  .extent = {extent.width, extent.height, 1},
                                  .usage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled,
                                  .debug_name = "Albedo image"});

  normal_image_.emplace(device_.get_allocator(), device_.get(),
                        ImageDesc{.format = vk::Format::eR16G16Snorm,
                                  .extent = {extent.width, extent.height, 1},
                                  .usage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled,
                                  .debug_name = "Normal image"});

  pbr_image_.emplace(device_.get_allocator(), device_.get(),
                     ImageDesc{.format = vk::Format::eR8G8B8A8Unorm,
                               .extent = {extent.width, extent.height, 1},
                               .usage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled,
                               .debug_name = "PBR image"});

  emissive_image_.emplace(
      device_.get_allocator(), device_.get(),
      ImageDesc{.format = vk::Format::eB10G11R11UfloatPack32,
                .extent = {extent.width, extent.height, 1},
                .usage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled,
                .debug_name = "Emissive image"});

  depth_image_.emplace(
      device_.get_allocator(), device_.get(),
      ImageDesc{.format = vk::Format::eD32Sfloat,
                .extent = {extent.width, extent.height, 1},
                .usage = vk::ImageUsageFlagBits::eDepthStencilAttachment | vk::ImageUsageFlagBits::eSampled,
                .debug_name = "Depth image"});

  fwrk::ImageImportInfo image_import_info{.type = VK_IMAGE_TYPE_2D,
                                          .size = {.width = extent.width, .height = extent.height, .depth = 1},
                                          .format = VK_FORMAT_R8G8B8A8_SRGB,
                                          .mip_count = 1,
                                          .layer_count = 1,
                                          .state = fwrk::PhysicalState::Undefined};

  if (albedo_id_) {
    context_.update_image(albedo_id_, image_import_info, albedo_image_->image());
  } else {
    albedo_id_ = context_.import_image(image_import_info, albedo_image_->image());
  }

  image_import_info.format = VK_FORMAT_R16G16_SNORM;
  if (normal_id_) {
    context_.update_image(normal_id_, image_import_info, normal_image_->image());
  } else {
    normal_id_ = context_.import_image(image_import_info, normal_image_->image());
  }

  image_import_info.format = VK_FORMAT_R8G8B8A8_UNORM;
  if (pbr_id_) {
    context_.update_image(pbr_id_, image_import_info, pbr_image_->image());
  } else {
    pbr_id_ = context_.import_image(image_import_info, pbr_image_->image());
  }

  image_import_info.format = VK_FORMAT_B10G11R11_UFLOAT_PACK32;
  if (emissive_id_) {
    context_.update_image(emissive_id_, image_import_info, emissive_image_->image());
  } else {
    emissive_id_ = context_.import_image(image_import_info, emissive_image_->image());
  }

  image_import_info.format = VK_FORMAT_D32_SFLOAT;
  if (depth_id_) {
    context_.update_image(depth_id_, image_import_info, depth_image_->image());
  } else {
    depth_id_ = context_.import_image(image_import_info, depth_image_->image());
  }
}

void Renderer3D::update_descriptors()
{
  DescriptorWriter writer{};

  for (uint32_t i = 0; i < texture_images_.size(); i++) {
    writer.add_image(2, Textures + i, vk::DescriptorType::eSampledImage, texture_images_.at(i).second.get(),
                     vk::ImageLayout::eShaderReadOnlyOptimal);
  }

  for (uint32_t i = 0; i < samplers_.size(); i++) {
    writer.add_image(3, TextureSamplers + i, vk::DescriptorType::eSampler, nullptr, vk::ImageLayout::eUndefined,
                     samplers_.at(i).get());
  }

  writer.update(device_.get(), bindless_descriptor_set_);
}

uint32_t Renderer3D::pack_texture_index(const MaterialInfo::Textures::Texture& texture)
{
  if (texture.index == UINT32_MAX) return UINT32_MAX;

  assert(texture.index < MaterialTextureMask);
  assert(texture.tex_coord <= 1);
  assert(texture.sampler <= MaterialSamplerMask);
  return pack_material_texture(texture.index, texture.tex_coord, texture.sampler);
}
