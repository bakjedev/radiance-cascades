#include "renderer_2d.hpp"

#include <SDL3/SDL_vulkan.h>
#include <cmath>

#include "backend/descriptor.hpp"
#include "backend/pipeline.hpp"
#include "src/event_dispatcher.hpp"
#include "src/resource/resource_manager.hpp"
#include "src/resource/types/shader_resource.hpp"
#include "src/window.hpp"

namespace {
  struct DrawPushConstant {
    uint32_t x;
    uint32_t y;
    uint32_t s;
    uint8_t material_id;
  };

  struct JFAPushConstant {
    uint32_t read_id;
    uint32_t write_id;
    uint32_t k;
  };

  struct ConvertPushConstant {
    uint32_t jfa_id;
  };

  struct SDFPushConstant {
    uint32_t jfa_id;
    uint32_t sdf_id;
  };

  struct CascadesPushConstant {
    uint32_t sdf_id;
    uint32_t cascades_id;
    uint32_t cascade_width;
    uint32_t cascade_height;
    uint32_t cascade;
    uint32_t base_probe_size;
    float base_spacing;
    uint32_t base_probe_dir_count;
    float base_length;
  };

  struct SpecialData {
    uint32_t image_width;
    uint32_t image_height;
  };

  struct Material {
    float color[3];
    float radiance;
  };

  struct DebugLineVertex {
    float color[3];
    float thickness;
    float pos[2];
  };

  struct GenDebugPushConstant {
    uint32_t vertex_id;
    uint32_t cascade_id;
    uint32_t cascade;
    uint32_t cascade_width;
    uint32_t cascade_height;
    uint32_t base_probe_size;
    float base_spacing;
    uint32_t base_probe_dir_count;
    float base_length;
  };
} // namespace

Renderer2D::Renderer2D(Window& window, EventDispatcher& event_dispatcher,
                       ResourceManager<ShaderResource>& resource_manager, FileSystem& file_system) :
    window_(window), event_dispatcher_(event_dispatcher), resource_manager_(resource_manager),
    file_system_(file_system), device_(instance_.get(), create_surface(window, instance_)),
    swapchain_(device_, {window.width(), window.height()},
               vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferDst),
    fwrk_allocator_(device_.get_allocator()), context_(device_.get(), frames_in_flight, fwrk_allocator_)
{
  for (Frame& frame: frames_) {
    create_frame(frame, device_);
  }

  submit_semaphores_.resize(swapchain_.image_count());
  constexpr vk::SemaphoreCreateInfo semaphore_create_info{};
  for (auto& semaphore: submit_semaphores_) {
    semaphore = device_.get().createSemaphoreUnique(semaphore_create_info);
  }

  event_dispatcher.listen<WindowResizeEvent>([this](const WindowResizeEvent& event) {
    swapchain_.recreate({static_cast<uint32_t>(event.width), static_cast<uint32_t>(event.height)});
    import_resources();
  });

  // ----------------------------------------
  // Images and Buffers
  // ----------------------------------------
  scene_image_.emplace(device_.get_allocator(), ImageDesc{}
                                                    .set_extent(config_.scene_size.width, config_.scene_size.height)
                                                    .set_format(vk::Format::eR8Uint)
                                                    .set_usage(vk::ImageUsageFlagBits::eStorage));
  scene_image_view_ = scene_image_->create_image_view(device_.get(), vk::ImageAspectFlagBits::eColor);

  material_buffer_.emplace(device_.get_allocator(),
                           BufferDesc{.alloc_flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT}
                               .set_size(sizeof(Material) * 256)
                               .set_usage(vk::BufferUsageFlagBits::eStorageBuffer));

  std::vector materials = {Material{}, Material{.color = {0.8f, 0.8f, 0.1f}, .radiance = 1.0f},
                           Material{.color = {0.1f, 0.8f, 0.5f}, .radiance = 1.0f}};
  void* material_data;
  vmaMapMemory(device_.get_allocator(), material_buffer_->allocation(), &material_data);
  memcpy(material_data, materials.data(), sizeof(Material) * materials.size());
  vmaUnmapMemory(device_.get_allocator(), material_buffer_->allocation());

  // ----------------------------------------
  // Descriptors
  // ----------------------------------------
  std::array sizes{vk::DescriptorPoolSize{vk::DescriptorType::eStorageImage, 100},
                   vk::DescriptorPoolSize{vk::DescriptorType::eStorageBuffer, 100}};
  descriptor_pool_ = create_descriptor_pool(device_.get(), sizes, 1);

  // Bindless
  {
    bindless_descriptor_set_layout_ = create_descriptor_set_layout(
        device_.get(), DescriptorSetLayoutDesc{}
                           .add_binding(0, vk::DescriptorType::eStorageImage, vk::ShaderStageFlagBits::eCompute,
                                        vk::DescriptorBindingFlagBits::ePartiallyBound, 100)
                           .add_binding(1, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eCompute,
                                        vk::DescriptorBindingFlagBits::ePartiallyBound, 100));

    bindless_descriptor_set_ = device_.get()
                                   .allocateDescriptorSets(vk::DescriptorSetAllocateInfo{}
                                                               .setDescriptorPool(*descriptor_pool_)
                                                               .setSetLayouts(*bindless_descriptor_set_layout_))
                                   .front();
  }

  DescriptorWriter{}
      .add_image(0, 0, vk::DescriptorType::eStorageImage, scene_image_view_.get(), vk::ImageLayout::eGeneral)
      .add_buffer(1, 0, vk::DescriptorType::eStorageBuffer, material_buffer_->buffer())
      .update(device_.get(), bindless_descriptor_set_);

  // ----------------------------------------
  // Pipelines
  // ----------------------------------------

  SpecialData special_data{.image_width = config_.scene_size.width, .image_height = config_.scene_size.height};
  std::array<vk::SpecializationMapEntry, 2> entries{{{0, offsetof(SpecialData, image_width), sizeof(uint32_t)},
                                                     {1, offsetof(SpecialData, image_height), sizeof(uint32_t)}}};
  vk::SpecializationInfo specialization_info{};
  specialization_info.setMapEntries(entries);
  specialization_info.setData<SpecialData>(special_data);

  // Draw pipeline
  {
    vk::PushConstantRange draw_push{vk::ShaderStageFlagBits::eCompute, 0, sizeof(DrawPushConstant)};

    auto draw_shader_resource =
        resource_manager_.create_from_file<ShaderResource>("draw.comp.spv", ShaderResourceLoader{&file_system_});

    draw_shader_module_ = create_shader_module(device_.get(), draw_shader_resource->code);

    draw_pipeline_layout_ =
        create_pipeline_layout(device_.get(), {&bindless_descriptor_set_layout_.get(), 1}, {&draw_push, 1});

    const ComputePipelineDesc draw_pipeline_desc{.module = draw_shader_module_.get(),
                                                 .specialization = &specialization_info,
                                                 .layout = draw_pipeline_layout_.get()};

    draw_pipeline_ = create_compute_pipeline(device_.get(), draw_pipeline_desc);
  }

  // Convert pipeline
  {
    vk::PushConstantRange convert_push{vk::ShaderStageFlagBits::eCompute, 0, sizeof(ConvertPushConstant)};

    auto convert_shader_resource =
        resource_manager_.create_from_file<ShaderResource>("convert.comp.spv", ShaderResourceLoader{&file_system_});

    convert_shader_module_ = create_shader_module(device_.get(), convert_shader_resource->code);

    convert_pipeline_layout_ =
        create_pipeline_layout(device_.get(), {&bindless_descriptor_set_layout_.get(), 1}, {&convert_push, 1});

    const ComputePipelineDesc convert_pipeline_desc{.module = convert_shader_module_.get(),
                                                    .specialization = &specialization_info,
                                                    .layout = convert_pipeline_layout_.get()};

    convert_pipeline_ = create_compute_pipeline(device_.get(), convert_pipeline_desc);
  }

  // JFA pipeline
  {
    vk::PushConstantRange jfa_push{vk::ShaderStageFlagBits::eCompute, 0, sizeof(JFAPushConstant)};

    auto jfa_shader_resource =
        resource_manager_.create_from_file<ShaderResource>("jfa.comp.spv", ShaderResourceLoader{&file_system_});

    jfa_shader_module_ = create_shader_module(device_.get(), jfa_shader_resource->code);

    jfa_pipeline_layout_ =
        create_pipeline_layout(device_.get(), {&bindless_descriptor_set_layout_.get(), 1}, {&jfa_push, 1});

    const ComputePipelineDesc jfa_pipeline_desc{.module = jfa_shader_module_.get(),
                                                .specialization = &specialization_info,
                                                .layout = jfa_pipeline_layout_.get()};

    jfa_pipeline_ = create_compute_pipeline(device_.get(), jfa_pipeline_desc);
  }

  // SDF pipeline
  {
    vk::PushConstantRange sdf_push{vk::ShaderStageFlagBits::eCompute, 0, sizeof(SDFPushConstant)};

    auto sdf_shader_resource =
        resource_manager_.create_from_file<ShaderResource>("sdf.comp.spv", ShaderResourceLoader{&file_system_});

    sdf_shader_module_ = create_shader_module(device_.get(), sdf_shader_resource->code);

    sdf_pipeline_layout_ =
        create_pipeline_layout(device_.get(), {&bindless_descriptor_set_layout_.get(), 1}, {&sdf_push, 1});

    const ComputePipelineDesc sdf_pipeline_desc{.module = sdf_shader_module_.get(),
                                                .specialization = &specialization_info,
                                                .layout = sdf_pipeline_layout_.get()};

    sdf_pipeline_ = create_compute_pipeline(device_.get(), sdf_pipeline_desc);
  }

  // Cascades pipeline
  {
    vk::PushConstantRange cascades_push{vk::ShaderStageFlagBits::eCompute, 0, sizeof(CascadesPushConstant)};

    auto cascades_shader_resource =
        resource_manager_.create_from_file<ShaderResource>("cascades.comp.spv", ShaderResourceLoader{&file_system_});

    cascades_shader_module_ = create_shader_module(device_.get(), cascades_shader_resource->code);

    cascades_pipeline_layout_ =
        create_pipeline_layout(device_.get(), {&bindless_descriptor_set_layout_.get(), 1}, {&cascades_push, 1});

    const ComputePipelineDesc cascades_pipeline_desc{.module = cascades_shader_module_.get(),
                                                     .specialization = &specialization_info,
                                                     .layout = cascades_pipeline_layout_.get()};

    cascades_pipeline_ = create_compute_pipeline(device_.get(), cascades_pipeline_desc);
  }

  // Generate debug lines pipeline
  {
    vk::PushConstantRange gen_debug_push{vk::ShaderStageFlagBits::eCompute, 0, sizeof(GenDebugPushConstant)};

    auto gen_debug_shader_resource =
        resource_manager_.create_from_file<ShaderResource>("gen_debug.comp.spv", ShaderResourceLoader{&file_system_});

    gen_debug_shader_module_ = create_shader_module(device_.get(), gen_debug_shader_resource->code);

    gen_debug_pipeline_layout_ =
        create_pipeline_layout(device_.get(), {&bindless_descriptor_set_layout_.get(), 1}, {&gen_debug_push, 1});

    const ComputePipelineDesc gen_debug_pipeline_desc{.module = gen_debug_shader_module_.get(),
                                                      .specialization = &specialization_info,
                                                      .layout = gen_debug_pipeline_layout_.get()};

    gen_debug_pipeline_ = create_compute_pipeline(device_.get(), gen_debug_pipeline_desc);
  }

  // Render debug lines pipeline
  {
    auto debug_vert_shader_resource =
        resource_manager_.create_from_file<ShaderResource>("debug.vert.spv", ShaderResourceLoader{&file_system_});
    auto debug_frag_shader_resource =
        resource_manager_.create_from_file<ShaderResource>("debug.frag.spv", ShaderResourceLoader{&file_system_});

    debug_vert_shader_module_ = create_shader_module(device_.get(), debug_vert_shader_resource->code);
    debug_frag_shader_module_ = create_shader_module(device_.get(), debug_frag_shader_resource->code);

    debug_pipeline_layout_ = create_pipeline_layout(device_.get(), {}, {});

    GraphicsPipelineDesc debug_pipeline_desc{};
    debug_pipeline_desc.stages.emplace_back(vk::ShaderStageFlagBits::eVertex, debug_vert_shader_module_.get(), "main",
                                            &specialization_info);
    debug_pipeline_desc.stages.emplace_back(vk::ShaderStageFlagBits::eFragment, debug_frag_shader_module_.get());

    debug_pipeline_desc.vertex_bindings.emplace_back(0, 32);

    debug_pipeline_desc.vertex_attributes.emplace_back(0, 0, vk::Format::eR32G32B32Sfloat,
                                                       offsetof(DebugLineVertex, color));
    debug_pipeline_desc.vertex_attributes.emplace_back(1, 0, vk::Format::eR32Sfloat,
                                                       offsetof(DebugLineVertex, thickness));
    debug_pipeline_desc.vertex_attributes.emplace_back(2, 0, vk::Format::eR32G32Sfloat, offsetof(DebugLineVertex, pos));

    debug_pipeline_desc.input_assembly.topology = vk::PrimitiveTopology::eLineList;

    debug_pipeline_desc.rasterization.lineWidth = 1.0;

    debug_pipeline_desc.depth_stencil.depthTestEnable = vk::False;
    debug_pipeline_desc.depth_stencil.depthWriteEnable = vk::False;

    debug_pipeline_desc.add_attachment(swapchain_.format());

    debug_pipeline_desc.layout = debug_pipeline_layout_.get();

    debug_pipeline_ = create_graphics_pipeline(device_.get(), debug_pipeline_desc);
  }

  // ----------------------------------------
  // Imports
  // ----------------------------------------
  fwrk::ImageImportInfo image_import_info{
      .type = VK_IMAGE_TYPE_2D,
      .size = {.width = config_.scene_size.width, .height = config_.scene_size.height, .depth = 1},
      .format = VK_FORMAT_R8_UINT,
      .state = fwrk::PhysicalState::Undefined};
  scene_image_import_ = context_.import_image(image_import_info, scene_image_->image());
  image_import_info.format = VK_FORMAT_R32G32_SINT;

  import_resources();
  swapchain_proxy_ = context_.create_proxy();
}

Renderer2D::~Renderer2D() { device_.get().waitIdle(); }

void Renderer2D::render()
{
  if (begin_frame()) {
    run_frame();
    end_frame();
  }
}
void Renderer2D::plot(const std::pair<float, float>& pos, const uint8_t material)
{
  const auto swp_w = swapchain_.extent().width;
  const auto swp_h = swapchain_.extent().height;
  const auto img_w = static_cast<float>(config_.scene_size.width);
  const auto img_h = static_cast<float>(config_.scene_size.height);

  const float scale = std::min(static_cast<float>(swp_w) / img_w, static_cast<float>(swp_h) / img_h);

  const auto dst_w = static_cast<uint32_t>(img_w * scale);
  const auto dst_h = static_cast<uint32_t>(img_h * scale);

  const uint32_t dst_off_x = (static_cast<uint32_t>(swp_w) - dst_w) / 2;
  const uint32_t dst_off_y = (static_cast<uint32_t>(swp_h) - dst_h) / 2;

  const float x_factor = (pos.first - static_cast<float>(dst_off_x)) / static_cast<float>(dst_w);
  const float y_factor = (pos.second - static_cast<float>(dst_off_y)) / static_cast<float>(dst_h);

  draw_material_ = material;
  draw_pos_ = {x_factor * img_w, y_factor * img_h};
}

bool Renderer2D::begin_frame()
{
  const Frame& frame = frames_.at(current_frame_);

  vk::Result result = device_.get().waitForFences(frame.in_flight.get(), vk::True, UINT64_MAX);
  if (result != vk::Result::eSuccess) {
    throw std::runtime_error("Failed to wait for fences: " + vk::to_string(result));
  }

  auto image_index = swapchain_.acquire_next_image(frame.image_available.get());
  if (!image_index) {
    swapchain_.recreate({window_.width(), window_.height()});
    import_resources();
    return false;
  }
  image_index_ = image_index.value();

  device_.get().resetFences(frame.in_flight.get());

  device_.get().resetCommandPool(frame.command_pool.get());
  frame.command_buffer->begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
  return true;
}

void Renderer2D::run_frame()
{
  const Frame& frame = frames_.at(current_frame_);
  fwrk::Graph& graph = context_.graph();

  if (should_compile_) {
    compile();
    should_compile_ = false;
  }

  context_.update_proxy(swapchain_proxy_, swapchain_imports_[image_index_]);

  graph.execute(frame.command_buffer.get(), current_frame_);
}

void Renderer2D::end_frame()
{
  const Frame& frame = frames_.at(current_frame_);
  frame.command_buffer->end();

  vk::SemaphoreSubmitInfo wait_semaphore{};
  wait_semaphore.setSemaphore(frame.image_available.get());
  wait_semaphore.setStageMask(vk::PipelineStageFlagBits2::eColorAttachmentOutput);

  vk::SemaphoreSubmitInfo signal_semaphore{};
  signal_semaphore.setSemaphore(submit_semaphores_.at(image_index_).get());
  signal_semaphore.setStageMask(vk::PipelineStageFlagBits2::eAllCommands);

  vk::CommandBufferSubmitInfo command_buffer_info{};
  command_buffer_info.setCommandBuffer(frame.command_buffer.get());

  vk::SubmitInfo2 submit_info{};
  submit_info.setWaitSemaphoreInfos(wait_semaphore);
  submit_info.setSignalSemaphoreInfos(signal_semaphore);
  submit_info.setCommandBufferInfos(command_buffer_info);

  device_.get_queue().submit2(submit_info, frame.in_flight.get());

  if (!swapchain_.present(device_.get_queue(), image_index_, submit_semaphores_.at(image_index_).get())) {
    swapchain_.recreate({window_.width(), window_.height()});
    import_resources();
  }

  current_frame_ = (current_frame_ + 1) % frames_in_flight;
}

void Renderer2D::compile()
{
  fwrk::Graph& graph = context_.graph();

  // ------------------
  // Transients
  // ------------------
  fwrk::ImageCreateInfo create_info{
      .type = VK_IMAGE_TYPE_2D,
      .size = {.width = config_.scene_size.width, .height = config_.scene_size.height, .depth = 1},
      .format = VK_FORMAT_R32G32_SINT,
      .flags = {},
      .mips = 1,
      .layers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = VK_IMAGE_USAGE_STORAGE_BIT};
  const fwrk::ResourceID jfa_1 = graph.create_image(create_info);
  const fwrk::ResourceID jfa_2 = graph.create_image(create_info);

  create_info.format = VK_FORMAT_R32_SFLOAT;
  create_info.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
  const fwrk::ResourceID sdf = graph.create_image(create_info);

  // kinda needs to be a multiple of 2.
  const auto base_probe_dir_count = static_cast<uint32_t>(std::round(360.0f / config_.cascades.base_interval));
  const auto probe_size = static_cast<uint32_t>(std::ceil(std::sqrt(base_probe_dir_count)));
  const auto probe_per_row =
      static_cast<uint32_t>(std::floor(static_cast<float>(config_.scene_size.width) / config_.cascades.base_spacing));
  const auto probe_per_col =
      static_cast<uint32_t>(std::floor(static_cast<float>(config_.scene_size.height) / config_.cascades.base_spacing));
  const auto cascade_width = probe_per_row * probe_size;
  const auto cascade_height = probe_per_col * probe_size;

  create_info.size.width = cascade_width;
  create_info.size.height = cascade_height;
  create_info.format = VK_FORMAT_R16G16B16A16_SFLOAT;
  create_info.layers = config_.cascades.cascades;
  create_info.usage = VK_IMAGE_USAGE_STORAGE_BIT;
  const fwrk::ResourceID cascades = graph.create_image(create_info);

  const auto base_probe_spacing = config_.cascades.base_spacing;
  const auto base_probe_length = config_.cascades.base_length;

  const fwrk::BufferCreateInfo debug_create_info{
      .size = sizeof(DebugLineVertex) * config_.debug_lines.max_debug_lines * 2,
      .flags = {},
      .usage = VK_BUFFER_USAGE_2_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT};
  const fwrk::ResourceID debug_lines_vertex = graph.create_buffer(debug_create_info);

  // ------------------
  // Passes
  // ------------------
  graph.add_compute_pass("Draw")
      .set_storage_image_write({.resource = {.id = scene_image_import_}})
      .set_storage_image_write({.resource = {.id = jfa_1}})
      .set_execute([this](vk::CommandBuffer cmd) { draw_pass(cmd); });

  graph.add_compute_pass("Convert")
      .set_storage_image_read({.resource = {.id = scene_image_import_}})
      .set_storage_image_write({.resource = {.id = jfa_1}})
      .set_execute([this](vk::CommandBuffer cmd) { convert_pass(cmd); });

  graph.add_compute_pass("JFA")
      .set_storage_image_write({.resource = {.id = jfa_1}})
      .set_storage_image_write({.resource = {.id = jfa_2}})
      .set_execute([this, jfa_1, jfa_2](vk::CommandBuffer cmd) { jfa_pass(cmd, jfa_1, jfa_2); });

  graph.add_compute_pass("SDF")
      .set_storage_image_read({.resource = {.id = jfa_2}})
      .set_storage_image_write({.resource = {.id = sdf}})
      .set_execute([this](vk::CommandBuffer cmd) { sdf_pass(cmd); });

  graph.add_compute_pass("Cascades")
      .set_storage_image_read({.resource = {.id = sdf}})
      .set_storage_image_write({.resource = {.id = cascades}})
      .set_execute([this, cascade_width, cascade_height, probe_size, base_probe_spacing, base_probe_dir_count,
                    base_probe_length](vk::CommandBuffer cmd) {
        cascades_pass(cmd, cascade_width, cascade_height, probe_size, base_probe_spacing, base_probe_dir_count,
                      base_probe_length);
      });

  // graph.add_compute_pass("Blit")
  //     .set_image_transfer_src({.resource = {.id = sdf}})
  //     .set_image_transfer_dst({.resource = {.id = swapchain_proxy_}})
  //     .set_execute([this, sdf](vk::CommandBuffer cmd) { blit_pass(cmd, sdf); });

  graph.add_compute_pass("Generate Debug Lines")
      .set_storage_buffer_write({.resource = {.id = debug_lines_vertex}})
      .set_storage_image_read({.resource = {.id = cascades}})
      .set_execute([this, cascade_width, cascade_height, probe_size, base_probe_spacing, base_probe_dir_count,
                    base_probe_length](vk::CommandBuffer cmd) {
        generate_debug_lines_pass(cmd, cascade_width, cascade_height, probe_size, base_probe_spacing,
                                  base_probe_dir_count, base_probe_length);
      });

  graph.add_graphics_pass("Debug lines")
      .set_color_attachment({.resource = {.id = swapchain_proxy_},
                             .load_op = fwrk::LoadOp::Clear,
                             .store_op = fwrk::StoreOp::Store,
                             .clear_value = fwrk::ClearValue{0.0f, 0.0f, 0.0f, 0.0f}})
      .set_vertex_buffer_input(
          {.resource = {.id = debug_lines_vertex}, .stages = VK_PIPELINE_STAGE_2_VERTEX_ATTRIBUTE_INPUT_BIT})
      .set_execute([this, debug_lines_vertex, cascade_width, cascade_height](vk::CommandBuffer cmd) {
        debug_lines_pass(cmd, debug_lines_vertex, cascade_width, cascade_height);
      });

  graph.set_image_end_state(swapchain_proxy_,
                            {VK_ACCESS_2_NONE, VK_PIPELINE_STAGE_2_NONE, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR});

  graph.compile();

  // Very ugly descriptor updates...
  vk::ImageSubresourceRange subresource_range{};
  subresource_range.setAspectMask(vk::ImageAspectFlagBits::eColor);
  subresource_range.setBaseMipLevel(0);
  subresource_range.setBaseArrayLayer(0);
  subresource_range.setLevelCount(vk::RemainingMipLevels);
  subresource_range.setLayerCount(vk::RemainingArrayLayers);
  const fwrk::ViewKey view_key{subresource_range, VK_IMAGE_VIEW_TYPE_2D};
  const fwrk::ViewKey array_view_key{subresource_range, VK_IMAGE_VIEW_TYPE_2D_ARRAY};

  DescriptorWriter{}
      .add_image(0, 1, vk::DescriptorType::eStorageImage, context_.acquire_image_view(jfa_1, view_key, 0),
                 vk::ImageLayout::eGeneral)
      .add_image(0, 2, vk::DescriptorType::eStorageImage, context_.acquire_image_view(jfa_2, view_key, 0),
                 vk::ImageLayout::eGeneral)
      .add_image(0, 3, vk::DescriptorType::eStorageImage, context_.acquire_image_view(jfa_1, view_key, 1),
                 vk::ImageLayout::eGeneral)
      .add_image(0, 4, vk::DescriptorType::eStorageImage, context_.acquire_image_view(jfa_2, view_key, 1),
                 vk::ImageLayout::eGeneral)
      .add_image(0, 5, vk::DescriptorType::eStorageImage, context_.acquire_image_view(sdf, view_key, 0),
                 vk::ImageLayout::eGeneral)
      .add_image(0, 6, vk::DescriptorType::eStorageImage, context_.acquire_image_view(sdf, view_key, 1),
                 vk::ImageLayout::eGeneral)
      .add_image(0, 7, vk::DescriptorType::eStorageImage, context_.acquire_image_view(cascades, array_view_key, 0),
                 vk::ImageLayout::eGeneral)
      .add_image(0, 8, vk::DescriptorType::eStorageImage, context_.acquire_image_view(cascades, array_view_key, 1),
                 vk::ImageLayout::eGeneral)
      .add_buffer(1, 1, vk::DescriptorType::eStorageBuffer, context_.get_raw_buffer(debug_lines_vertex, 0))
      .add_buffer(1, 2, vk::DescriptorType::eStorageBuffer, context_.get_raw_buffer(debug_lines_vertex, 1))
      .update(device_.get(), bindless_descriptor_set_);
}

void Renderer2D::draw_pass(vk::CommandBuffer cmd)
{
  if (draw_material_ == 0) return;

  cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, draw_pipeline_layout_.get(), 0, 1, &bindless_descriptor_set_,
                         0, nullptr);
  cmd.bindPipeline(vk::PipelineBindPoint::eCompute, draw_pipeline_.get());

  const DrawPushConstant push_constant{
      .x = draw_pos_.first, .y = draw_pos_.second, .s = config_.drawing.size, .material_id = draw_material_};

  cmd.pushConstants(draw_pipeline_layout_.get(), vk::ShaderStageFlagBits::eCompute, 0, sizeof(DrawPushConstant),
                    &push_constant);

  const uint32_t gs = (config_.drawing.size + 7) / 8;
  cmd.dispatch(gs, gs, 1);

  draw_material_ = 0;
}

void Renderer2D::convert_pass(vk::CommandBuffer cmd)
{
  cmd.bindPipeline(vk::PipelineBindPoint::eCompute, convert_pipeline_.get());
  cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, convert_pipeline_layout_.get(), 0, 1,
                         &bindless_descriptor_set_, 0, nullptr);

  const ConvertPushConstant push_constant{
      .jfa_id = 1 + current_frame_ * 2,
  };

  cmd.pushConstants(convert_pipeline_layout_.get(), vk::ShaderStageFlagBits::eCompute, 0, sizeof(ConvertPushConstant),
                    &push_constant);

  const uint32_t gx = (config_.scene_size.width + 7) / 8;
  const uint32_t gy = (config_.scene_size.height + 7) / 8;
  cmd.dispatch(gx, gy, 1);
}

void Renderer2D::jfa_pass(vk::CommandBuffer cmd, const fwrk::ResourceID jfa_1, const fwrk::ResourceID jfa_2)
{
  cmd.bindPipeline(vk::PipelineBindPoint::eCompute, jfa_pipeline_.get());
  cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, jfa_pipeline_layout_.get(), 0, 1, &bindless_descriptor_set_,
                         0, nullptr);


  auto barrier = vk::ImageMemoryBarrier2{}
                     .setSrcStageMask(vk::PipelineStageFlagBits2::eComputeShader)
                     .setSrcAccessMask(vk::AccessFlagBits2::eShaderStorageWrite)
                     .setDstStageMask(vk::PipelineStageFlagBits2::eComputeShader)
                     .setDstAccessMask(vk::AccessFlagBits2::eShaderStorageRead)
                     .setOldLayout(vk::ImageLayout::eGeneral)
                     .setNewLayout(vk::ImageLayout::eGeneral)
                     .setSrcQueueFamilyIndex(vk::QueueFamilyIgnored)
                     .setDstQueueFamilyIndex(vk::QueueFamilyIgnored)
                     .setSubresourceRange({vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1});

  const uint32_t gx = (config_.scene_size.width + 7) / 8;
  const uint32_t gy = (config_.scene_size.height + 7) / 8;

  bool use_jfa_1 = true;
  const auto k_start = std::max(config_.scene_size.width, config_.scene_size.height);
  const auto k_count = k_start <= 1 ? 0 : static_cast<uint32_t>(std::bit_width(k_start - 1));
  for (uint32_t i = 1; i <= k_count; i++) {
    const JFAPushConstant push_constant{.read_id = (use_jfa_1 ? 1 : 2) + current_frame_ * 2,
                                        .write_id = (use_jfa_1 ? 2 : 1) + current_frame_ * 2,
                                        .k = k_start >> i};
    cmd.pushConstants(jfa_pipeline_layout_.get(), vk::ShaderStageFlagBits::eCompute, 0, sizeof(JFAPushConstant),
                      &push_constant);

    cmd.dispatch(gx, gy, 1);

    if (i > 1) {
      barrier.setImage(use_jfa_1 ? context_.get_raw_image(jfa_1, current_frame_)
                                 : context_.get_raw_image(jfa_2, current_frame_));
      cmd.pipelineBarrier2(vk::DependencyInfo{}.setImageMemoryBarriers(barrier));
    }
    use_jfa_1 = !use_jfa_1;
  }
}

void Renderer2D::sdf_pass(vk::CommandBuffer cmd)
{
  cmd.bindPipeline(vk::PipelineBindPoint::eCompute, sdf_pipeline_.get());
  cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, sdf_pipeline_layout_.get(), 0, 1, &bindless_descriptor_set_,
                         0, nullptr);

  const SDFPushConstant push_constant{
      .jfa_id = (1 + current_frame_) * 2,
      .sdf_id = 5 + current_frame_,
  };

  cmd.pushConstants(sdf_pipeline_layout_.get(), vk::ShaderStageFlagBits::eCompute, 0, sizeof(SDFPushConstant),
                    &push_constant);

  const uint32_t gx = (config_.scene_size.width + 7) / 8;
  const uint32_t gy = (config_.scene_size.height + 7) / 8;
  cmd.dispatch(gx, gy, 1);
}

void Renderer2D::cascades_pass(vk::CommandBuffer cmd, const uint32_t cascade_width, const uint32_t cascade_height,
                               const uint32_t probe_size, const float base_probe_spacing,
                               const uint32_t base_probe_dir_count, const float base_probe_length)
{
  cmd.bindPipeline(vk::PipelineBindPoint::eCompute, cascades_pipeline_.get());
  cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, cascades_pipeline_layout_.get(), 0, 1,
                         &bindless_descriptor_set_, 0, nullptr);

  CascadesPushConstant push_constant{.sdf_id = 5 + current_frame_,
                                     .cascades_id = 7 + current_frame_,
                                     .cascade_width = cascade_width,
                                     .cascade_height = cascade_height,
                                     .cascade = 3,
                                     .base_probe_size = probe_size,
                                     .base_spacing = base_probe_spacing,
                                     .base_probe_dir_count = base_probe_dir_count,
                                     .base_length = base_probe_length};

  const uint32_t gx = (cascade_width + 7) / 8;
  const uint32_t gy = (cascade_height + 7) / 8;

  for (uint32_t i = 0; i < config_.cascades.cascades; i++) {
    push_constant.cascade = i;

    cmd.pushConstants(cascades_pipeline_layout_.get(), vk::ShaderStageFlagBits::eCompute, 0,
                      sizeof(CascadesPushConstant), &push_constant);


    cmd.dispatch(gx, gy, 1);
  }
}

void Renderer2D::blit_pass(vk::CommandBuffer cmd, const fwrk::ResourceID sdf)
{
  const auto swp_w = swapchain_.extent().width;
  const auto swp_h = swapchain_.extent().height;
  const auto img_w = static_cast<float>(config_.scene_size.width);
  const auto img_h = static_cast<float>(config_.scene_size.height);

  const float scale = std::min(static_cast<float>(swp_w) / img_w, static_cast<float>(swp_h) / img_h);

  const auto dst_w = static_cast<int32_t>(img_w * scale);
  const auto dst_h = static_cast<int32_t>(img_h * scale);

  const int32_t dst_off_x = (static_cast<int32_t>(swp_w) - dst_w) / 2;
  const int32_t dst_off_y = (static_cast<int32_t>(swp_h) - dst_h) / 2;

  const vk::ImageBlit blit{{vk::ImageAspectFlagBits::eColor, 0, 0, 1},
                           {{{0, 0, 0}, {static_cast<int32_t>(img_w), static_cast<int32_t>(img_h), 1}}},
                           {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
                           {{{dst_off_x, dst_off_y, 0}, {dst_off_x + dst_w, dst_off_y + dst_h, 1}}}};

  cmd.blitImage(context_.get_raw_image(sdf), vk::ImageLayout::eTransferSrcOptimal, swapchain_.image(image_index_),
                vk::ImageLayout::eTransferDstOptimal, 1, &blit, vk::Filter::eLinear);
}

void Renderer2D::generate_debug_lines_pass(vk::CommandBuffer cmd, const uint32_t cascade_width,
                                           const uint32_t cascade_height, const uint32_t probe_size,
                                           const float spacing, const uint32_t probe_dir_count, const float length)
{
  cmd.bindPipeline(vk::PipelineBindPoint::eCompute, gen_debug_pipeline_.get());
  cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, gen_debug_pipeline_layout_.get(), 0, 1,
                         &bindless_descriptor_set_, 0, nullptr);

  const uint32_t gx = (cascade_width + 7) / 8;
  const uint32_t gy = (cascade_height + 7) / 8;

  GenDebugPushConstant push_constant{.vertex_id = 1 + current_frame_,
                                     .cascade_id = 7 + current_frame_,
                                     .cascade = 0,
                                     .cascade_width = cascade_width,
                                     .cascade_height = cascade_height,
                                     .base_probe_size = probe_size,
                                     .base_spacing = spacing,
                                     .base_probe_dir_count = probe_dir_count,
                                     .base_length = length};

  for (uint32_t i = 0; i < config_.cascades.cascades; i++) {
    push_constant.cascade = i;
    cmd.pushConstants(gen_debug_pipeline_layout_.get(), vk::ShaderStageFlagBits::eCompute, 0,
                      sizeof(GenDebugPushConstant), &push_constant);

    cmd.dispatch(gx, gy, 1);
  }
}

void Renderer2D::debug_lines_pass(vk::CommandBuffer cmd, const fwrk::ResourceID debug_lines_vertex,
                                  const uint32_t cascade_width, const uint32_t cascade_height)
{
  cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, debug_pipeline_.get());
  constexpr vk::DeviceSize offset = 0;
  const auto vertex_buffer = vk::Buffer{context_.get_raw_buffer(debug_lines_vertex)};
  cmd.bindVertexBuffers(0, 1, &vertex_buffer, &offset);

  const auto swp_w = static_cast<float>(swapchain_.extent().width);
  const auto swp_h = static_cast<float>(swapchain_.extent().height);
  const auto img_w = static_cast<float>(config_.scene_size.width);
  const auto img_h = static_cast<float>(config_.scene_size.height);

  const float scale = std::min(swp_w / img_w, swp_h / img_h);

  const float dst_w = img_w * scale;
  const float dst_h = img_h * scale;

  const float dst_off_x = (swp_w - dst_w) / 2;
  const float dst_off_y = (swp_h - dst_h) / 2;

  const vk::Viewport viewport{dst_off_x, dst_off_y, dst_w, dst_h};
  cmd.setViewport(0, 1, &viewport);

  const vk::Rect2D scissor{vk::Offset2D{0, 0}, vk::Extent2D{swapchain_.extent().width, swapchain_.extent().height}};
  cmd.setScissor(0, 1, &scissor);

  const uint32_t vertex_count = cascade_width * cascade_height * 2 * config_.cascades.cascades;

  cmd.draw(vertex_count, 1, 0, 0);
}


void Renderer2D::import_resources()
{
  swapchain_imports_.resize(swapchain_.image_count());

  const fwrk::ImageImportInfo swapchain_image_info{.type = VK_IMAGE_TYPE_2D,
                                                   .size = {swapchain_.extent().width, swapchain_.extent().height, 1},
                                                   .format = static_cast<VkFormat>(swapchain_.format()),
                                                   .state = fwrk::PhysicalState::Undefined};

  for (uint32_t i = 0; i < swapchain_.image_count(); i++) {
    fwrk::ResourceID& res = swapchain_imports_[i];
    if (res) {
      context_.update_image(res, swapchain_image_info, swapchain_.image(i));
    } else {
      res = context_.import_image(swapchain_image_info, swapchain_.image(i), "Swapchain image");
    }
  }
}

VkSurfaceKHR Renderer2D::create_surface(const Window& window, const Instance& instance)
{
  VkSurfaceKHR surface = VK_NULL_HANDLE;
  if (!SDL_Vulkan_CreateSurface(window.get(), instance.get(), nullptr, &surface)) {
    throw std::runtime_error("Failed to create Vulkan Surface with SDL");
  }
  return surface;
}
