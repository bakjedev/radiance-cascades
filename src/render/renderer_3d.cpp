#include "renderer_3d.hpp"

#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/matrix_transform.hpp>

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

  struct ShadowPushConstant {
    uint32_t cascade;
    uint32_t cascades_id;
    uint32_t geometries_id;
  };

  struct CompositePushConstant {
    uint32_t composite_id;
    uint32_t frame_data_id;
    uint32_t cascades_id;
    uint32_t lights_data_id;
    uint32_t shadow_id;
    uint32_t shadow_sampler_id;
    uint32_t lights_id;
    uint32_t albedo_id;
    uint32_t normal_id;
    uint32_t pbr_id;
    uint32_t emissive_id;
    uint32_t depth_id;
  };

  struct BlitPushConstant {
    uint32_t composite_id;
    bool aces;
  };

  enum TextureDescriptors {
    AlbedoTexture = 0,
    NormalTexture,
    PBRTexture,
    EmissiveTexture,
    DepthTexture,
    ShadowTexture,
    CompositeTexture,
    Textures,
  };

  enum BufferDescriptors {
    GeometriesBuffer = 0,
    MaterialBuffer,
    LightsBuffer,
  };

  enum SamplerDescriptors {
    DefaultSampler = 0,
    ShadowSampler,
    TextureSamplers,
  };

  enum ImageDescriptors {
    CompositeImage = 0,
  };

  enum UBODescriptors {
    FrameDataUBO = 0,
    CascadedShadowsUBO,
    LightsDataUBO,
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

  lights_buffer_.emplace(device_.get_allocator(), device_.get(),
                         BufferDesc{.size = sizeof(Light) * max_lights * frames_in_flight,
                                    .usage = vk::BufferUsageFlagBits::eStorageBuffer,
                                    .memory_usage = VMA_MEMORY_USAGE_AUTO,
                                    .alloc_flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                                                   VMA_ALLOCATION_CREATE_MAPPED_BIT,
                                    .debug_name = "Lights buffer"});

  const auto ubo_alignment = device_.get_physical().getProperties().limits.minUniformBufferOffsetAlignment;
  frame_data_size_ = (sizeof(FrameData) + ubo_alignment - 1) & ~(ubo_alignment - 1);
  cascades_size_ = (sizeof(CascadedShadows) + ubo_alignment - 1) & ~(ubo_alignment - 1);
  lights_data_size_ = (sizeof(LightsData) + ubo_alignment - 1) & ~(ubo_alignment - 1);
  constexpr auto frame_data_base = static_cast<vk::DeviceSize>(0);
  cascades_base_ = frame_data_size_ * frames_in_flight;
  lights_data_base_ = cascades_base_ + cascades_size_ * frames_in_flight;

  ubo_buffer_.emplace(
      device_.get_allocator(), device_.get(),
      BufferDesc{}
          .set_size(frame_data_size_ * frames_in_flight + cascades_size_ * frames_in_flight +
                    lights_data_size_ * frames_in_flight)
          .set_usage(vk::BufferUsageFlagBits::eUniformBuffer)
          .set_allocate_flags(VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT)
          .set_debug_name("UBOs"));

  shadow_image_.emplace(
      device_.get_allocator(), device_.get(),
      ImageDesc{}
          .set_format(vk::Format::eD32Sfloat)
          .set_extent(CascadedShadows::Resolution, CascadedShadows::Resolution)
          .set_layers(CascadedShadows::Count)
          .set_usage(vk::ImageUsageFlagBits::eDepthStencilAttachment | vk::ImageUsageFlagBits::eSampled)
          .set_debug_name("Shadow Image"));
  shadow_view_ =
      shadow_image_->create_image_view(device_.get(), vk::ImageAspectFlagBits::eDepth, vk::ImageViewType::e2DArray);
  shadow_id_ = context_.import_image({.type = VK_IMAGE_TYPE_2D,
                                      .size = {CascadedShadows::Resolution, CascadedShadows::Resolution, 1},
                                      .format = VK_FORMAT_D32_SFLOAT,
                                      .mip_count = 1,
                                      .layer_count = CascadedShadows::Count,
                                      .state = fwrk::PhysicalState::Undefined},
                                     shadow_image_->image(), "Shadow Image");

  vk::SamplerCreateInfo sampler_info{};
  sampler_info.setAddressModeU(vk::SamplerAddressMode::eClampToBorder);
  sampler_info.setAddressModeV(vk::SamplerAddressMode::eClampToBorder);
  sampler_info.setMinFilter(vk::Filter::eLinear);
  sampler_info.setMagFilter(vk::Filter::eLinear);
  sampler_info.setBorderColor(vk::BorderColor::eFloatOpaqueWhite);
  sampler_info.setCompareEnable(vk::True);
  sampler_info.setCompareOp(vk::CompareOp::eLessOrEqual);
  shadow_sampler_ = device_.get().createSamplerUnique(sampler_info);

  std::vector<vk::DebugUtilsObjectNameInfoEXT> debug_object_names;
  debug_object_names.reserve(1);

  auto handle_to_uint64_t = []<typename T>(T handle) -> uint64_t {
    return reinterpret_cast<uint64_t>(static_cast<T::CType>(handle));
  };

  {
    vk::PushConstantRange raster_push{vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0,
                                      sizeof(RasterPushConstant)};

    const auto raster_shader = create_info.load_shader("assets:/shaders/3d/raster.spv");
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

    raster_pipeline_desc.rasterization.cullMode = vk::CullModeFlagBits::eBack;

    raster_pipeline_ = create_graphics_pipeline(device_.get(), raster_pipeline_desc);

    debug_object_names.emplace_back(vk::ObjectType::ePipeline, handle_to_uint64_t(raster_pipeline_.get()),
                                    "Raster Pipeline");
  }

  {
    vk::PushConstantRange shadow_push{vk::ShaderStageFlagBits::eVertex, 0, sizeof(ShadowPushConstant)};

    const auto shadow_shader = create_info.load_shader("assets:/shaders/3d/shadow.spv");
    shadow_shader_module_ = create_shader_module(device_.get(), shadow_shader);

    shadow_pipeline_layout_ =
        create_pipeline_layout(device_.get(), {&bindless_descriptor_set_layout_.get(), 1}, {&shadow_push, 1});

    GraphicsPipelineDesc shadow_pipeline_desc{};
    shadow_pipeline_desc.stages.emplace_back(vk::ShaderStageFlagBits::eVertex, shadow_shader_module_.get(), "main");

    shadow_pipeline_desc.vertex_bindings = {
        {0, sizeof(Vertex), vk::VertexInputRate::eVertex},
    };

    shadow_pipeline_desc.vertex_attributes = {
        {0, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, position)},
    };

    shadow_pipeline_desc.depth_format = vk::Format::eD32Sfloat;

    shadow_pipeline_desc.depth_stencil.depthTestEnable = vk::True;
    shadow_pipeline_desc.depth_stencil.depthWriteEnable = vk::True;
    shadow_pipeline_desc.depth_stencil.depthCompareOp = vk::CompareOp::eLess;

    shadow_pipeline_desc.layout = shadow_pipeline_layout_.get();

    shadow_pipeline_desc.rasterization.depthClampEnable = vk::True;
    shadow_pipeline_desc.rasterization.depthBiasEnable = vk::True;
    shadow_pipeline_desc.rasterization.cullMode = vk::CullModeFlagBits::eNone;

    shadow_pipeline_desc.dynamic_states.push_back(vk::DynamicState::eDepthBias);

    shadow_pipeline_ = create_graphics_pipeline(device_.get(), shadow_pipeline_desc);

    debug_object_names.emplace_back(vk::ObjectType::ePipeline, handle_to_uint64_t(shadow_pipeline_.get()),
                                    "Shadow Pipeline");
  }

  {
    vk::PushConstantRange composite_push{vk::ShaderStageFlagBits::eCompute, 0, sizeof(CompositePushConstant)};

    const auto composite_shader = create_info.load_shader("assets:/shaders/3d/composite.spv");
    composite_shader_module_ = create_shader_module(device_.get(), composite_shader);

    composite_pipeline_layout_ =
        create_pipeline_layout(device_.get(), {&bindless_descriptor_set_layout_.get(), 1}, {&composite_push, 1});

    const ComputePipelineDesc composite_pipeline_desc{.module = composite_shader_module_.get(),
                                                      .layout = composite_pipeline_layout_.get()};

    composite_pipeline_ = create_compute_pipeline(device_.get(), composite_pipeline_desc);

    debug_object_names.emplace_back(vk::ObjectType::ePipeline, handle_to_uint64_t(composite_pipeline_.get()),
                                    "Composite Pipeline");
  }

  {
    vk::PushConstantRange blit_push{vk::ShaderStageFlagBits::eFragment, 0, sizeof(BlitPushConstant)};

    const auto blit_shader = create_info.load_shader("assets:/shaders/blit.spv");
    blit_shader_module_ = create_shader_module(device_.get(), blit_shader);

    blit_pipeline_layout_ =
        create_pipeline_layout(device_.get(), {&bindless_descriptor_set_layout_.get(), 1}, {&blit_push, 1});

    GraphicsPipelineDesc blit_pipeline_desc{};
    blit_pipeline_desc.stages.emplace_back(vk::ShaderStageFlagBits::eVertex, blit_shader_module_.get(), "vs_main");
    blit_pipeline_desc.stages.emplace_back(vk::ShaderStageFlagBits::eFragment, blit_shader_module_.get(), "fs_main");

    blit_pipeline_desc.depth_stencil.depthTestEnable = vk::False;
    blit_pipeline_desc.depth_stencil.depthWriteEnable = vk::False;

    blit_pipeline_desc.add_attachment(swapchain_.format());

    blit_pipeline_desc.layout = blit_pipeline_layout_.get();

    blit_pipeline_ = create_graphics_pipeline(device_.get(), blit_pipeline_desc);

    debug_object_names.emplace_back(vk::ObjectType::ePipeline, handle_to_uint64_t(blit_pipeline_.get()),
                                    "Blit Pipeline");
  }

  for (auto& info: debug_object_names) {
    if (device_.get().setDebugUtilsObjectNameEXT(&info) != vk::Result::eSuccess) {
      throw std::runtime_error("Failed to set object name on pipeline");
    }
  }

  create_render_targets(create_info.initial_extent);

  const auto ubo_buf = ubo_buffer_->buffer();

  DescriptorWriter writer{};

  writer.add_buffer(1, GeometriesBuffer, vk::DescriptorType::eStorageBuffer, geometries_buffer_->buffer())
      .add_buffer(1, LightsBuffer, vk::DescriptorType::eStorageBuffer, lights_buffer_->buffer())
      .add_image(3, DefaultSampler, vk::DescriptorType::eSampler, nullptr, vk::ImageLayout::eUndefined, sampler_.get())
      .add_image(3, ShadowSampler, vk::DescriptorType::eSampler, nullptr, vk::ImageLayout::eUndefined,
                 shadow_sampler_.get());

  for (uint32_t i = 0; i < frames_in_flight; i++) {
    writer.add_buffer(4, i, vk::DescriptorType::eUniformBuffer, ubo_buf, frame_data_base + frame_data_size_ * i,
                      frame_data_size_);

    writer.add_buffer(4, frames_in_flight + i, vk::DescriptorType::eUniformBuffer, ubo_buf,
                      cascades_base_ + cascades_size_ * i, cascades_size_);

    writer.add_buffer(4, 2 * frames_in_flight + i, vk::DescriptorType::eUniformBuffer, ubo_buf,
                      lights_data_base_ + lights_data_size_ * i, lights_data_size_);
  }

  writer.update(device_.get(), bindless_descriptor_set_);

  frame_data_.sun_dir = glm::normalize(glm::vec3(0.3f, 1.0f, 0.2f));
  frame_data_.sun_color = glm::vec3(1.0f, 0.95f, 0.9f) * 3.0f;
  frame_data_.ambient = glm::vec3(0.15f, 0.18f, 0.25f);
  frame_data_.ambient_indirect = glm::vec3(0.05f, 0.04f, 0.03f);
}

Renderer3D::~Renderer3D() { device_.get().waitIdle(); }

void Renderer3D::render(RenderInfo info)
{
  info_ = std::move(info);

  const vk::Extent2D extent = {info_.window_size.first, info_.window_size.second};
  if (needs_recreate_ || extent != swapchain_.extent()) {
    device_.get().waitIdle();
    create_render_targets(extent);
  }

  if (!begin_frame(extent)) {
    return;
  }

  if (upload_materials_) {
    upload_materials();
    upload_materials_ = false;
  }

  upload_geometry();

  upload_lights();

  update_frame_data();

  update_cascades();

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

  create_info.setMaxLod(vk::LodClampNone);
  create_info.setAnisotropyEnable(vk::True);
  create_info.setMaxAnisotropy(device_.get_physical().getProperties().limits.maxSamplerAnisotropy);

  switch (min_filter) {
    case Filter::Nearest:
      create_info.setMinFilter(vk::Filter::eNearest);
      create_info.setMipmapMode(vk::SamplerMipmapMode::eNearest);
      create_info.setMaxLod(0.25f);
      break;
    case Filter::Linear:
      create_info.setMinFilter(vk::Filter::eLinear);
      create_info.setMipmapMode(vk::SamplerMipmapMode::eNearest);
      create_info.setMaxLod(0.25f);
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
                          pack_texture_index(material.textures.emissive), material.alpha_mode, material.alpha_cutoff);
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

void Renderer3D::submit_geometry(const GeometrySubmission& submission) { geometry_submissions_.push_back(submission); }

void Renderer3D::submit_light(const PointLightSubmission& submission)
{
  point_lights_.emplace_back(submission.position, submission.range, glm::vec3(0.0f), 0.0f, submission.radiance, 1.0f);
}

void Renderer3D::submit_light(const SpotLightSubmission& submission)
{
  assert(submission.cos_inner >= submission.cos_outer);
  const auto angle_size = 1.0f / std::max(0.001f, submission.cos_inner - submission.cos_outer);
  spot_lights_.emplace_back(submission.position, submission.range, glm::normalize(submission.direction), angle_size,
                            submission.radiance, -submission.cos_outer * angle_size);
}

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

  for (uint32_t i = 0; i < CascadedShadows::Count; i++) {
    graph.add_graphics_pass(std::format("Shadow Cascade {}", i))
        .set_depth_attachment({.resource = {.id = shadow_id_},
                               .load_op = fwrk::LoadOp::Clear,
                               .store_op = fwrk::StoreOp::Store,
                               .clear_value = {1.0f},
                               .base_layer = i})
        .set_execute([this, i](vk::CommandBuffer cmd) { shadow_pass(cmd, i); });
  }

  graph.add_compute_pass("Composite")
      .set_image_read({.resource = {.id = albedo_id_}})
      .set_image_read({.resource = {.id = normal_id_}})
      .set_image_read({.resource = {.id = pbr_id_}})
      .set_image_read({.resource = {.id = emissive_id_}})
      .set_image_read({.resource = {.id = depth_id_}})
      .set_image_read({.resource = {.id = shadow_id_}})
      .set_storage_image_write({.resource = {.id = composite_id_}})
      .set_execute([this](vk::CommandBuffer cmd) { composite_pass(cmd); });

  graph.add_graphics_pass("Blit")
      .set_image_read({.resource = {.id = composite_id_}})
      .set_color_attachment(
          {.resource = {.id = swapchain_proxy_}, .load_op = fwrk::LoadOp::DontCare, .store_op = fwrk::StoreOp::Store})
      .set_execute([this](vk::CommandBuffer cmd) { blit_pass(cmd); });

  graph.add_graphics_pass("ImGui")
      .set_color_attachment({
          .resource = {.id = swapchain_proxy_},
          .load_op = fwrk::LoadOp::Load,
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

void Renderer3D::shadow_pass(vk::CommandBuffer cmd, const uint32_t cascade) const
{
  cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, shadow_pipeline_.get());
  cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, shadow_pipeline_layout_.get(), 0, 1,
                         &bindless_descriptor_set_, 0, nullptr);

  constexpr vk::Viewport viewport{
      0.0f, 0.0f, static_cast<float>(CascadedShadows::Resolution), static_cast<float>(CascadedShadows::Resolution),
      0.0f, 1.0f};
  cmd.setViewport(0, 1, &viewport);
  constexpr vk::Rect2D scissor{vk::Offset2D{0, 0},
                               vk::Extent2D{CascadedShadows::Resolution, CascadedShadows::Resolution}};
  cmd.setScissor(0, 1, &scissor);

  const float filter_texels = cascades_.filter_radius_per_texel[static_cast<int>(cascade)];
  cmd.setDepthBias(0.0f, (filter_texels + 2.0f) / static_cast<float>(CascadedShadows::Resolution),
                   filter_texels + 1.0f);

  const ShadowPushConstant push_constant{.cascade = cascade,
                                         .cascades_id = CascadedShadowsUBO * frames_in_flight + current_frame_,
                                         .geometries_id = GeometriesBuffer};

  cmd.pushConstants(shadow_pipeline_layout_.get(), vk::ShaderStageFlagBits::eVertex, 0, sizeof(ShadowPushConstant),
                    &push_constant);

  constexpr VkDeviceSize offset = 0;
  const auto vertex_buffer = vertex_buffer_->buffer();
  const auto index_buffer = index_buffer_->buffer();
  cmd.bindVertexBuffers(0, 1, &vertex_buffer, &offset);
  cmd.bindIndexBuffer(index_buffer, offset, vk::IndexType::eUint32);

  constexpr auto stride = sizeof(vk::DrawIndexedIndirectCommand);
  cmd.drawIndexedIndirect(indirect_buffer_->buffer(), stride * max_geometries * current_frame_, geometry_starts_[1],
                          stride);
}

void Renderer3D::composite_pass(vk::CommandBuffer cmd) const
{
  cmd.bindPipeline(vk::PipelineBindPoint::eCompute, composite_pipeline_.get());
  cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, composite_pipeline_layout_.get(), 0, 1,
                         &bindless_descriptor_set_, 0, nullptr);

  const CompositePushConstant push_constant{.composite_id = CompositeImage,
                                            .frame_data_id = FrameDataUBO * frames_in_flight + current_frame_,
                                            .cascades_id = CascadedShadowsUBO * frames_in_flight + current_frame_,
                                            .lights_data_id = LightsDataUBO * frames_in_flight + current_frame_,
                                            .shadow_id = ShadowTexture,
                                            .shadow_sampler_id = ShadowSampler,
                                            .lights_id = LightsBuffer,
                                            .albedo_id = AlbedoTexture,
                                            .normal_id = NormalTexture,
                                            .pbr_id = PBRTexture,
                                            .emissive_id = EmissiveTexture,
                                            .depth_id = DepthTexture};

  cmd.pushConstants(composite_pipeline_layout_.get(), vk::ShaderStageFlagBits::eCompute, 0,
                    sizeof(CompositePushConstant), &push_constant);

  const uint32_t gx = (info_.window_size.first + 7) / 8;
  const uint32_t gy = (info_.window_size.second + 7) / 8;
  cmd.dispatch(gx, gy, 1);
}

void Renderer3D::blit_pass(vk::CommandBuffer cmd) const
{
  cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, blit_pipeline_.get());
  cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, blit_pipeline_layout_.get(), 0, 1, &bindless_descriptor_set_,
                         0, nullptr);

  const vk::Viewport viewport{0, 0, static_cast<float>(info_.window_size.first),
                              static_cast<float>(info_.window_size.second)};
  cmd.setViewport(0, 1, &viewport);

  const vk::Rect2D scissor{vk::Offset2D{0, 0}, vk::Extent2D{info_.window_size.first, info_.window_size.second}};
  cmd.setScissor(0, 1, &scissor);

  constexpr BlitPushConstant push_constant{
      .composite_id = CompositeTexture,
      .aces = true,
  };

  cmd.pushConstants(blit_pipeline_layout_.get(), vk::ShaderStageFlagBits::eFragment, 0, sizeof(BlitPushConstant),
                    &push_constant);

  cmd.draw(3, 1, 0, 0);
}

void Renderer3D::upload_geometry()
{
  constexpr auto indirect_region = sizeof(vk::DrawIndexedIndirectCommand) * max_geometries;
  constexpr auto geometries_region = sizeof(Geometry) * max_geometries;

  auto* indirect_mapping = reinterpret_cast<vk::DrawIndexedIndirectCommand*>(
      indirect_buffer_->get_mapped_data<std::byte>() + indirect_region * current_frame_);
  auto* geometries_mapping = reinterpret_cast<Geometry*>(geometries_buffer_->get_mapped_data<std::byte>() +
                                                         geometries_region * current_frame_);

  geometry_count_ = static_cast<uint32_t>(geometry_submissions_.size());

  auto get_alpha_mode = [&](const GeometrySubmission& submission) {
    return materials_[submission.material.id].alpha_mode;
  };

  std::array<uint32_t, 3> counts{};
  for (const auto& submission: geometry_submissions_) counts[get_alpha_mode(submission)]++;

  std::array<uint32_t, 3> cursor{};
  uint32_t start = 0;
  for (uint32_t i = 0; i < 3; i++) {
    geometry_starts_[i] = start;
    cursor[i] = start;
    start += counts[i];
  }

  for (const auto& submission: geometry_submissions_) {
    const auto& [range, material, transform] = submission;
    const uint32_t i = cursor[get_alpha_mode(submission)]++;
    indirect_mapping[i] = vk::DrawIndexedIndirectCommand{}
                              .setIndexCount(range.index_count)
                              .setInstanceCount(1)
                              .setFirstIndex(range.first_index)
                              .setVertexOffset(range.vertex_offset)
                              .setFirstInstance(current_frame_ * max_geometries + i);
    geometries_mapping[i] = Geometry{.transform = transform, .material = material};
  }
  geometry_submissions_.clear();
}

void Renderer3D::upload_lights()
{
  constexpr auto lights_region = sizeof(Light) * max_lights;

  auto* lights_mapping =
      reinterpret_cast<Light*>(lights_buffer_->get_mapped_data<std::byte>() + lights_region * current_frame_);

  const auto point_count = static_cast<uint32_t>(std::min<size_t>(point_lights_.size(), max_point_lights));
  const auto spot_count = static_cast<uint32_t>(std::min<size_t>(spot_lights_.size(), max_spot_lights));

  std::memcpy(lights_mapping, point_lights_.data(), point_count * sizeof(Light));
  std::memcpy(lights_mapping + max_point_lights, spot_lights_.data(), spot_count * sizeof(Light));

  point_lights_.clear();
  spot_lights_.clear();

  lights_data_.point_light_count = point_count;
  lights_data_.spot_light_count = spot_count;
  lights_data_.point_lights_base = max_lights * current_frame_;
  lights_data_.spot_lights_base = max_lights * current_frame_ + max_point_lights;

  std::memcpy(ubo_buffer_->get_mapped_data<std::byte>() + lights_data_base_ + lights_data_size_ * current_frame_,
              &lights_data_, sizeof(LightsData));
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

void Renderer3D::update_frame_data()
{
  frame_data_.inv_view_proj = glm::inverse(info_.projection * info_.view);
  frame_data_.camera_pos = -glm::vec3(info_.view[3]) * glm::mat3(info_.view);
  std::memcpy(ubo_buffer_->get_mapped_data<std::byte>() + frame_data_size_ * current_frame_, &frame_data_,
              sizeof(FrameData));
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

  composite_image_.emplace(device_.get_allocator(), device_.get(),
                           ImageDesc{.format = vk::Format::eR16G16B16A16Sfloat,
                                     .extent = {extent.width, extent.height, 1},
                                     .usage = vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled,
                                     .debug_name = "Composite image"});

  albedo_view_ = albedo_image_->create_image_view(device_.get(), vk::ImageAspectFlagBits::eColor);
  normal_view_ = normal_image_->create_image_view(device_.get(), vk::ImageAspectFlagBits::eColor);
  pbr_view_ = pbr_image_->create_image_view(device_.get(), vk::ImageAspectFlagBits::eColor);
  emissive_view_ = emissive_image_->create_image_view(device_.get(), vk::ImageAspectFlagBits::eColor);
  depth_view_ = depth_image_->create_image_view(device_.get(), vk::ImageAspectFlagBits::eDepth);

  composite_view_ = composite_image_->create_image_view(device_.get(), vk::ImageAspectFlagBits::eColor);

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

  image_import_info.format = VK_FORMAT_R16G16B16A16_SFLOAT;
  if (composite_id_) {
    context_.update_image(composite_id_, image_import_info, composite_image_->image());
  } else {
    composite_id_ = context_.import_image(image_import_info, composite_image_->image());
  }
}

void Renderer3D::update_descriptors()
{
  DescriptorWriter writer{};

  writer
      .add_image(0, CompositeImage, vk::DescriptorType::eStorageImage, composite_view_.get(), vk::ImageLayout::eGeneral)
      .add_image(2, AlbedoTexture, vk::DescriptorType::eSampledImage, albedo_view_.get(),
                 vk::ImageLayout::eShaderReadOnlyOptimal)
      .add_image(2, NormalTexture, vk::DescriptorType::eSampledImage, normal_view_.get(),
                 vk::ImageLayout::eShaderReadOnlyOptimal)
      .add_image(2, PBRTexture, vk::DescriptorType::eSampledImage, pbr_view_.get(),
                 vk::ImageLayout::eShaderReadOnlyOptimal)
      .add_image(2, EmissiveTexture, vk::DescriptorType::eSampledImage, emissive_view_.get(),
                 vk::ImageLayout::eShaderReadOnlyOptimal)
      .add_image(2, DepthTexture, vk::DescriptorType::eSampledImage, depth_view_.get(),
                 vk::ImageLayout::eShaderReadOnlyOptimal)
      .add_image(2, ShadowTexture, vk::DescriptorType::eSampledImage, shadow_view_.get(),
                 vk::ImageLayout::eShaderReadOnlyOptimal)
      .add_image(2, CompositeTexture, vk::DescriptorType::eSampledImage, composite_view_.get(),
                 vk::ImageLayout::eShaderReadOnlyOptimal);


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

void Renderer3D::update_cascades()
{
  const glm::mat4 camera_world = glm::inverse(info_.view);
  const auto camera_pos = glm::vec3(camera_world[3]);
  const glm::vec3 camera_forward = -glm::normalize(glm::vec3(camera_world[2]));

  const float tan_half_fov_x = 1.0f / std::abs(info_.projection[0][0]);
  const float tan_half_fov_y = 1.0f / std::abs(info_.projection[1][1]);

  const glm::vec4 near_center_view = glm::inverse(info_.projection) * glm::vec4(0.0f, 0.0f, 1.0f, 1.0f);
  const float near_plane = -near_center_view.z / near_center_view.w;

  const float corner_offset_sq = tan_half_fov_x * tan_half_fov_x + tan_half_fov_y * tan_half_fov_y;
  const float corner_offset = std::sqrt(corner_offset_sq);

  const glm::vec3 up =
      std::abs(frame_data_.sun_dir.y) > 0.99f ? glm::vec3(0.0f, 0.0f, 1.0f) : glm::vec3(0.0f, 1.0f, 0.0f);
  const glm::mat4 light_view = glm::lookAt(glm::vec3(0.0f), -frame_data_.sun_dir, up);

  float slice_near = near_plane;
  for (uint32_t i = 0; i < CascadedShadows::Count; i++) {
    const float split_fraction = static_cast<float>(i + 1) / static_cast<float>(CascadedShadows::Count);
    const float uniform_split = near_plane + (CascadedShadows::Distance - near_plane) * split_fraction;
    const float log_split = near_plane * std::pow(CascadedShadows::Distance / near_plane, split_fraction);
    const float slice_far = glm::mix(uniform_split, log_split, CascadedShadows::Lambda);

    float sphere_center_depth;
    float sphere_radius;
    const float far_clamp_threshold = (slice_far - slice_near) / (slice_far + slice_near);
    if (corner_offset_sq >= far_clamp_threshold) {
      sphere_center_depth = slice_far;
      sphere_radius = slice_far * corner_offset;
    } else {
      sphere_center_depth = 0.5f * (slice_far + slice_near) * (1.0f + corner_offset_sq);
      const float near_corner_depth = slice_near - sphere_center_depth;
      const float near_corner_distance_sq = slice_near * slice_near * corner_offset_sq;
      sphere_radius = std::sqrt(near_corner_depth * near_corner_depth + near_corner_distance_sq);
    }

    const glm::vec3 sphere_center = camera_pos + camera_forward * sphere_center_depth;

    auto sphere_center_light_view = glm::vec3(light_view * glm::vec4(sphere_center, 1.0f));

    const float world_units_per_texel = 2.0f * sphere_radius / static_cast<float>(CascadedShadows::Resolution);
    sphere_center_light_view.x = std::floor(sphere_center_light_view.x / world_units_per_texel) * world_units_per_texel;
    sphere_center_light_view.y = std::floor(sphere_center_light_view.y / world_units_per_texel) * world_units_per_texel;

    const float box_left = sphere_center_light_view.x - sphere_radius;
    const float box_right = sphere_center_light_view.x + sphere_radius;
    const float box_bottom = sphere_center_light_view.y - sphere_radius;
    const float box_top = sphere_center_light_view.y + sphere_radius;
    const float box_near = sphere_center_light_view.z + sphere_radius;
    const float box_far = sphere_center_light_view.z - sphere_radius;

    const glm::mat4 light_to_shadow_clip = glm::ortho(box_left, box_right, box_bottom, box_top, -box_near, -box_far);

    cascades_.world_to_shadow_clip[i] = light_to_shadow_clip * light_view;
    cascades_.world_units_per_texel[static_cast<int>(i)] = world_units_per_texel;
    cascades_.filter_radius_per_texel[static_cast<int>(i)] =
        glm::clamp(CascadedShadows::FilterRadius / world_units_per_texel, 1.0f, 4.0f);

    slice_near = slice_far;
  }

  cascades_.inv_res = 1.0 / CascadedShadows::Resolution;
  cascades_.blend = CascadedShadows::Blend;

  std::memcpy(ubo_buffer_->get_mapped_data<std::byte>() + cascades_base_ + cascades_size_ * current_frame_, &cascades_,
              sizeof(CascadedShadows));
}

uint32_t Renderer3D::pack_texture_index(const MaterialInfo::Textures::Texture& texture)
{
  if (texture.index == UINT32_MAX) return UINT32_MAX;

  assert(texture.index < MaterialTextureMask);
  assert(texture.tex_coord <= 1);
  assert(texture.sampler <= MaterialSamplerMask);
  return pack_material_texture(texture.index, texture.tex_coord, texture.sampler);
}
