#include "renderer_2d.hpp"

#include <SDL3/SDL_vulkan.h>
#include <cmath>

#include "backend/descriptor.hpp"
#include "backend/pipeline.hpp"
#include "imgui.h"
#include "imgui_impl_sdl3.h"
#include "imgui_impl_vulkan.h"
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
    uint32_t jfa_id;
    uint32_t read_idx;
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

  struct UBOData {
    uint32_t image_width;
    uint32_t image_height;
  };

  struct MergePushConstant {
    uint32_t cascade_id;
    uint32_t cascade;
    uint32_t cascade_width;
    uint32_t cascade_height;
    uint32_t base_probe_size;
  };

  struct LightPushConstant {
    uint32_t cascade_id;
    uint32_t light_id;
    uint32_t sdf_id;
    float spacing;
    uint32_t probe_size;
  };

  struct PullPushConstant {
    uint32_t read_id;
    uint32_t write_id;
    uint32_t width;
    uint32_t height;
  };

  struct CompositePushConstant {
    uint32_t composite_id;
    uint32_t light_id;
    uint32_t sdf_id;
    uint32_t scene_id;
    float rim;
  };

  struct BlitPushConstant {
    uint32_t composite_id;
  };

  enum StorageDescriptors {
    SceneImage = 0,
    JFAImage,
    SDFImage,
    CascadesImage,
    CompositeImage,
    LightImage,
  };

  enum SampledDescriptors { CompositeTexture = 0 };
} // namespace

Renderer2D::Renderer2D(Window& window, EventDispatcher& event_dispatcher,
                       ResourceManager<ShaderResource>& resource_manager, FileSystem& file_system) :
    window_(window), event_dispatcher_(event_dispatcher), resource_manager_(resource_manager),
    file_system_(file_system), device_(instance_.get(), create_surface(window, instance_)),
    swapchain_(device_, {window.width(), window.height()}, vk::ImageUsageFlagBits::eColorAttachment),
    context_(instance_.get(), device_.get(), frames_in_flight, empty_allocator_, true)
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
    import_swapchain_images();
  });

  // ----------------------------------------
  // Buffers
  // ----------------------------------------
  material_buffer_.emplace(device_.get_allocator(),
                           BufferDesc{.alloc_flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT}
                               .set_size(sizeof(Material) * 256)
                               .set_usage(vk::BufferUsageFlagBits::eStorageBuffer));

  materials_.push_back(Material{.color = {1.0f, 1.0f, 1.0f}, .radiance = {0.0f, 0.0f, 0.0f}});
  materials_.push_back(Material{.color = {1.0f, 1.0f, 1.0f}, .radiance = {2.0f, 2.0f, 2.0f}});
  materials_.push_back(Material{.color = {1.0f, 1.0f, 0.0f}, .radiance = {1.5f, 1.5f, 0.0f}});
  materials_.push_back(Material{.color = {0.8f, 0.8f, 0.8f}, .radiance = {0.0f, 0.0f, 0.0f}});
  materials_.push_back(Material{.color = {1.0f, 1.0f, 1.0f}, .radiance = {1.0f, 0.1f, 0.1f}});
  materials_.push_back(Material{.color = {0.0f, 0.0f, 1.0f}, .radiance = {0.0f, 0.0f, 0.0f}});

  void* material_data;
  vmaMapMemory(device_.get_allocator(), material_buffer_->allocation(), &material_data);
  memcpy(material_data, materials_.data(), sizeof(Material) * materials_.size());
  vmaUnmapMemory(device_.get_allocator(), material_buffer_->allocation());


  ubo_.emplace(device_.get_allocator(),
               BufferDesc{.alloc_flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT}
                   .set_size(sizeof(UBOData))
                   .set_usage(vk::BufferUsageFlagBits::eUniformBuffer));

  // ----------------------------------------
  // Sampler
  // ----------------------------------------
  constexpr vk::SamplerCreateInfo sampler_info{};
  sampler_ = device_.get().createSamplerUnique(sampler_info);

  // ----------------------------------------
  // Descriptors
  // ----------------------------------------
  std::array sizes{vk::DescriptorPoolSize{vk::DescriptorType::eStorageImage, 100},
                   vk::DescriptorPoolSize{vk::DescriptorType::eStorageBuffer, 100},
                   vk::DescriptorPoolSize{vk::DescriptorType::eSampledImage, 100},
                   vk::DescriptorPoolSize{vk::DescriptorType::eSampler, 100},
                   vk::DescriptorPoolSize{vk::DescriptorType::eUniformBuffer, 100}};
  descriptor_pool_ = create_descriptor_pool(device_.get(), sizes, 1);

  // Bindless
  {
    bindless_descriptor_set_layout_ = create_descriptor_set_layout(
        device_.get(), DescriptorSetLayoutDesc{}
                           .add_binding(0, vk::DescriptorType::eStorageImage, vk::ShaderStageFlagBits::eAll,
                                        vk::DescriptorBindingFlagBits::ePartiallyBound, 100)
                           .add_binding(1, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eAll,
                                        vk::DescriptorBindingFlagBits::ePartiallyBound, 100)
                           .add_binding(2, vk::DescriptorType::eSampledImage, vk::ShaderStageFlagBits::eAll,
                                        vk::DescriptorBindingFlagBits::ePartiallyBound, 100)
                           .add_binding(3, vk::DescriptorType::eSampler, vk::ShaderStageFlagBits::eAll,
                                        vk::DescriptorBindingFlagBits::ePartiallyBound, 100)
                           .add_binding(4, vk::DescriptorType::eUniformBuffer, vk::ShaderStageFlagBits::eAll,
                                        vk::DescriptorBindingFlagBits::ePartiallyBound, 100));

    bindless_descriptor_set_ = device_.get()
                                   .allocateDescriptorSets(vk::DescriptorSetAllocateInfo{}
                                                               .setDescriptorPool(*descriptor_pool_)
                                                               .setSetLayouts(*bindless_descriptor_set_layout_))
                                   .front();
  }

  DescriptorWriter{}
      .add_buffer(1, 0, vk::DescriptorType::eStorageBuffer, material_buffer_->buffer())
      .add_image(3, 0, vk::DescriptorType::eSampler, nullptr, vk::ImageLayout::eUndefined, sampler_.get())
      .add_buffer(4, 0, vk::DescriptorType::eUniformBuffer, ubo_->buffer())
      .update(device_.get(), bindless_descriptor_set_);

  // ----------------------------------------
  // ImGui
  // ----------------------------------------
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGui_ImplSDL3_InitForVulkan(window_.get());

  ImGui_ImplVulkan_InitInfo imgui_vulkan_info = {};
  imgui_vulkan_info.ApiVersion = vk::ApiVersion13;
  imgui_vulkan_info.Instance = instance_.get();
  imgui_vulkan_info.PhysicalDevice = device_.get_physical();
  imgui_vulkan_info.Device = device_.get();
  imgui_vulkan_info.QueueFamily = device_.get_queue_family();
  imgui_vulkan_info.Queue = device_.get_queue();
  imgui_vulkan_info.DescriptorPoolSize = IMGUI_IMPL_VULKAN_MINIMUM_SAMPLED_IMAGE_POOL_SIZE;
  imgui_vulkan_info.MinImageCount = 2;
  imgui_vulkan_info.ImageCount = swapchain_.image_count();
  imgui_vulkan_info.UseDynamicRendering = true;
  imgui_vulkan_info.PipelineInfoMain.PipelineRenderingCreateInfo.sType =
      VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR;
  imgui_vulkan_info.PipelineInfoMain.PipelineRenderingCreateInfo.colorAttachmentCount = 1;
  std::array imgui_format{static_cast<VkFormat>(swapchain_.format())};
  imgui_vulkan_info.PipelineInfoMain.PipelineRenderingCreateInfo.pColorAttachmentFormats = imgui_format.data();

  ImGui_ImplVulkan_Init(&imgui_vulkan_info);

  // ----------------------------------------
  // Pipelines
  // ----------------------------------------

  // Draw pipeline
  {
    vk::PushConstantRange draw_push{vk::ShaderStageFlagBits::eCompute, 0, sizeof(DrawPushConstant)};

    auto draw_shader_resource =
        resource_manager_.create_from_file<ShaderResource>("draw.slang.spv", ShaderResourceLoader{&file_system_});

    draw_shader_module_ = create_shader_module(device_.get(), draw_shader_resource->code);

    draw_pipeline_layout_ =
        create_pipeline_layout(device_.get(), {&bindless_descriptor_set_layout_.get(), 1}, {&draw_push, 1});

    const ComputePipelineDesc draw_pipeline_desc{.module = draw_shader_module_.get(),
                                                 .layout = draw_pipeline_layout_.get()};

    draw_pipeline_ = create_compute_pipeline(device_.get(), draw_pipeline_desc);
  }

  // Convert pipeline
  {
    vk::PushConstantRange convert_push{vk::ShaderStageFlagBits::eCompute, 0, sizeof(ConvertPushConstant)};

    auto convert_shader_resource =
        resource_manager_.create_from_file<ShaderResource>("convert.slang.spv", ShaderResourceLoader{&file_system_});

    convert_shader_module_ = create_shader_module(device_.get(), convert_shader_resource->code);

    convert_pipeline_layout_ =
        create_pipeline_layout(device_.get(), {&bindless_descriptor_set_layout_.get(), 1}, {&convert_push, 1});

    const ComputePipelineDesc convert_pipeline_desc{.module = convert_shader_module_.get(),
                                                    .layout = convert_pipeline_layout_.get()};

    convert_pipeline_ = create_compute_pipeline(device_.get(), convert_pipeline_desc);
  }

  // JFA pipeline
  {
    vk::PushConstantRange jfa_push{vk::ShaderStageFlagBits::eCompute, 0, sizeof(JFAPushConstant)};

    auto jfa_shader_resource =
        resource_manager_.create_from_file<ShaderResource>("jfa.slang.spv", ShaderResourceLoader{&file_system_});

    jfa_shader_module_ = create_shader_module(device_.get(), jfa_shader_resource->code);

    jfa_pipeline_layout_ =
        create_pipeline_layout(device_.get(), {&bindless_descriptor_set_layout_.get(), 1}, {&jfa_push, 1});

    const ComputePipelineDesc jfa_pipeline_desc{.module = jfa_shader_module_.get(),
                                                .layout = jfa_pipeline_layout_.get()};

    jfa_pipeline_ = create_compute_pipeline(device_.get(), jfa_pipeline_desc);
  }

  // SDF pipeline
  {
    vk::PushConstantRange sdf_push{vk::ShaderStageFlagBits::eCompute, 0, sizeof(SDFPushConstant)};

    auto sdf_shader_resource =
        resource_manager_.create_from_file<ShaderResource>("sdf.slang.spv", ShaderResourceLoader{&file_system_});

    sdf_shader_module_ = create_shader_module(device_.get(), sdf_shader_resource->code);

    sdf_pipeline_layout_ =
        create_pipeline_layout(device_.get(), {&bindless_descriptor_set_layout_.get(), 1}, {&sdf_push, 1});

    const ComputePipelineDesc sdf_pipeline_desc{.module = sdf_shader_module_.get(),
                                                .layout = sdf_pipeline_layout_.get()};

    sdf_pipeline_ = create_compute_pipeline(device_.get(), sdf_pipeline_desc);
  }

  // Cascades pipeline
  {
    vk::PushConstantRange cascades_push{vk::ShaderStageFlagBits::eCompute, 0, sizeof(CascadesPushConstant)};

    auto cascades_shader_resource =
        resource_manager_.create_from_file<ShaderResource>("cascades.slang.spv", ShaderResourceLoader{&file_system_});

    cascades_shader_module_ = create_shader_module(device_.get(), cascades_shader_resource->code);

    cascades_pipeline_layout_ =
        create_pipeline_layout(device_.get(), {&bindless_descriptor_set_layout_.get(), 1}, {&cascades_push, 1});

    const ComputePipelineDesc cascades_pipeline_desc{.module = cascades_shader_module_.get(),
                                                     .layout = cascades_pipeline_layout_.get()};

    cascades_pipeline_ = create_compute_pipeline(device_.get(), cascades_pipeline_desc);
  }

  // Merge pipeline
  {
    vk::PushConstantRange merge_push{vk::ShaderStageFlagBits::eCompute, 0, sizeof(MergePushConstant)};

    auto merge_shader_resource =
        resource_manager_.create_from_file<ShaderResource>("merge.slang.spv", ShaderResourceLoader{&file_system_});

    merge_shader_module_ = create_shader_module(device_.get(), merge_shader_resource->code);

    merge_pipeline_layout_ =
        create_pipeline_layout(device_.get(), {&bindless_descriptor_set_layout_.get(), 1}, {&merge_push, 1});

    const ComputePipelineDesc merge_pipeline_desc{.module = merge_shader_module_.get(),
                                                  .layout = merge_pipeline_layout_.get()};

    merge_pipeline_ = create_compute_pipeline(device_.get(), merge_pipeline_desc);
  }

  // Light pipeline
  {
    vk::PushConstantRange light_push{vk::ShaderStageFlagBits::eCompute, 0, sizeof(LightPushConstant)};

    auto light_shader_resource =
        resource_manager_.create_from_file<ShaderResource>("light.slang.spv", ShaderResourceLoader{&file_system_});

    light_shader_module_ = create_shader_module(device_.get(), light_shader_resource->code);

    light_pipeline_layout_ =
        create_pipeline_layout(device_.get(), {&bindless_descriptor_set_layout_.get(), 1}, {&light_push, 1});

    const ComputePipelineDesc light_pipeline_desc{.module = light_shader_module_.get(),
                                                  .layout = light_pipeline_layout_.get()};

    light_pipeline_ = create_compute_pipeline(device_.get(), light_pipeline_desc);
  }

  // Push pull pipelines
  {
    vk::PushConstantRange pull_push_push{vk::ShaderStageFlagBits::eCompute, 0, sizeof(PullPushConstant)};

    auto pull_push_shader_resource =
        resource_manager_.create_from_file<ShaderResource>("pull_push.slang.spv", ShaderResourceLoader{&file_system_});

    pull_push_shader_module_ = create_shader_module(device_.get(), pull_push_shader_resource->code);

    pull_push_pipeline_layout_ =
        create_pipeline_layout(device_.get(), {&bindless_descriptor_set_layout_.get(), 1}, {&pull_push_push, 1});

    const ComputePipelineDesc pull_pipeline_desc{
        .module = pull_push_shader_module_.get(), .entry = "pull", .layout = pull_push_pipeline_layout_.get()};
    pull_pipeline_ = create_compute_pipeline(device_.get(), pull_pipeline_desc);

    const ComputePipelineDesc push_pipeline_desc{
        .module = pull_push_shader_module_.get(), .entry = "push", .layout = pull_push_pipeline_layout_.get()};
    push_pipeline_ = create_compute_pipeline(device_.get(), push_pipeline_desc);
  }

  // Composite pipeline
  {
    vk::PushConstantRange composite_push{vk::ShaderStageFlagBits::eCompute, 0, sizeof(CompositePushConstant)};

    auto composite_shader_resource =
        resource_manager_.create_from_file<ShaderResource>("composite.slang.spv", ShaderResourceLoader{&file_system_});

    composite_shader_module_ = create_shader_module(device_.get(), composite_shader_resource->code);

    composite_pipeline_layout_ =
        create_pipeline_layout(device_.get(), {&bindless_descriptor_set_layout_.get(), 1}, {&composite_push, 1});

    const ComputePipelineDesc composite_pipeline_desc{.module = composite_shader_module_.get(),
                                                      .layout = composite_pipeline_layout_.get()};

    composite_pipeline_ = create_compute_pipeline(device_.get(), composite_pipeline_desc);
  }

  // Blit pipeline
  {
    vk::PushConstantRange blit_push{vk::ShaderStageFlagBits::eFragment, 0, sizeof(BlitPushConstant)};

    auto blit_shader_resource =
        resource_manager_.create_from_file<ShaderResource>("blit.slang.spv", ShaderResourceLoader{&file_system_});

    blit_shader_module_ = create_shader_module(device_.get(), blit_shader_resource->code);

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
  }

  // ----------------------------------------
  // Imports
  // ----------------------------------------
  import_swapchain_images();
  swapchain_proxy_ = context_.create_proxy();
}

Renderer2D::~Renderer2D()
{
  device_.get().waitIdle();
  ImGui_ImplVulkan_Shutdown();
  ImGui_ImplSDL3_Shutdown();
  ImGui::DestroyContext();
}

void Renderer2D::render()
{
  if (begin_frame()) {
    run_frame();
    end_frame();
  }
}
void Renderer2D::plot(const std::pair<float, float>& pos)
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

  draw_pos_ = {static_cast<uint32_t>(x_factor * img_w), static_cast<uint32_t>(y_factor * img_h)};
  should_draw = true;
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
    import_swapchain_images();
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
    import_swapchain_images();
  }

  current_frame_ = (current_frame_ + 1) % frames_in_flight;
}

void Renderer2D::compile()
{
  fwrk::Graph& graph = context_.graph();

  create_images();
  create_cascade_image();
  import_images();
  update_descriptor();

  // ------------------
  // Passes
  // ------------------
  graph.add_compute_pass("Draw")
      .set_storage_image_write({.resource = {.id = scene_id_}})
      .set_storage_image_write({.resource = {.id = jfa_id_}})
      .set_execute([this](vk::CommandBuffer cmd) { draw_pass(cmd); });

  graph.add_compute_pass("Convert")
      .set_storage_image_read({.resource = {.id = scene_id_}})
      .set_storage_image_write({.resource = {.id = jfa_id_}})
      .set_execute([this](vk::CommandBuffer cmd) { convert_pass(cmd); });

  graph.add_compute_pass("JFA")
      .set_storage_image_write({.resource = {.id = jfa_id_}})
      .set_execute([this](vk::CommandBuffer cmd) { jfa_pass(cmd); });

  graph.add_compute_pass("SDF")
      .set_storage_image_read({.resource = {.id = jfa_id_}})
      .set_storage_image_write({.resource = {.id = sdf_id_}})
      .set_execute([this](vk::CommandBuffer cmd) { sdf_pass(cmd); });

  graph.add_compute_pass("Cascades")
      .set_storage_image_read({.resource = {.id = sdf_id_}})
      .set_storage_image_write({.resource = {.id = cascades_id_}})
      .set_execute([this](vk::CommandBuffer cmd) { cascades_pass(cmd); });

  graph.add_compute_pass("Merge")
      .set_storage_image_write({.resource = {.id = cascades_id_}})
      .set_execute([this](vk::CommandBuffer cmd) { merge_pass(cmd); });

  graph.add_compute_pass("Light")
      .set_storage_image_read({.resource = {.id = cascades_id_}})
      .set_storage_image_write({.resource = {.id = light_id_}, .level_count = 1})
      .set_execute([this](vk::CommandBuffer cmd) { light_pass(cmd); });

  graph.add_compute_pass("Pull Push")
      .set_storage_image_write({.resource = {.id = light_id_}})
      .set_execute([this](vk::CommandBuffer cmd) { pull_push_pass(cmd); });

  graph.add_compute_pass("Composite")
      .set_storage_image_read({.resource = {.id = light_id_}, .level_count = 1})
      .set_storage_image_read({.resource = {.id = sdf_id_}})
      .set_storage_image_read({.resource = {.id = scene_id_}})
      .set_storage_image_write({.resource = {.id = composite_id_}})
      .set_execute([this](vk::CommandBuffer cmd) { composite_pass(cmd); });

  graph.add_graphics_pass("Blit")
      .set_image_read({.resource = {.id = composite_id_}})
      .set_color_attachment({.resource = {.id = swapchain_proxy_},
                             .load_op = fwrk::LoadOp::Clear,
                             .store_op = fwrk::StoreOp::Store,
                             .clear_value = {.r = 0.0f, .g = 0.0f, .b = 0.0f, .a = 0.0}})
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

void Renderer2D::draw_pass(vk::CommandBuffer cmd)
{
  if (!should_draw) return;

  cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, draw_pipeline_layout_.get(), 0, 1, &bindless_descriptor_set_,
                         0, nullptr);
  cmd.bindPipeline(vk::PipelineBindPoint::eCompute, draw_pipeline_.get());

  const DrawPushConstant push_constant{
      .x = draw_pos_.first, .y = draw_pos_.second, .s = config_.drawing.size, .material_id = draw_material_};

  cmd.pushConstants(draw_pipeline_layout_.get(), vk::ShaderStageFlagBits::eCompute, 0, sizeof(DrawPushConstant),
                    &push_constant);

  const uint32_t gs = (config_.drawing.size + 7) / 8;
  cmd.dispatch(gs, gs, 1);

  should_draw = false;
}

void Renderer2D::convert_pass(vk::CommandBuffer cmd)
{
  cmd.bindPipeline(vk::PipelineBindPoint::eCompute, convert_pipeline_.get());
  cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, convert_pipeline_layout_.get(), 0, 1,
                         &bindless_descriptor_set_, 0, nullptr);

  const ConvertPushConstant push_constant{
      .jfa_id = JFAImage,
  };

  cmd.pushConstants(convert_pipeline_layout_.get(), vk::ShaderStageFlagBits::eCompute, 0, sizeof(ConvertPushConstant),
                    &push_constant);

  const uint32_t gx = (config_.scene_size.width + 7) / 8;
  const uint32_t gy = (config_.scene_size.height + 7) / 8;
  cmd.dispatch(gx, gy, 1);
}

void Renderer2D::jfa_pass(vk::CommandBuffer cmd)
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
                     .setSubresourceRange({vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1})
                     .setImage(jfa_image_->image());

  const uint32_t gx = (config_.scene_size.width + 7) / 8;
  const uint32_t gy = (config_.scene_size.height + 7) / 8;

  bool use_jfa_2 = false;
  const auto k_start = std::max(config_.scene_size.width, config_.scene_size.height);
  const auto k_count = k_start <= 1 ? 0 : static_cast<uint32_t>(std::bit_width(k_start - 1));
  for (uint32_t i = 1; i <= k_count; i++) {
    const JFAPushConstant push_constant{
        .jfa_id = JFAImage, .read_idx = static_cast<uint32_t>(use_jfa_2), .k = k_start >> i};
    cmd.pushConstants(jfa_pipeline_layout_.get(), vk::ShaderStageFlagBits::eCompute, 0, sizeof(JFAPushConstant),
                      &push_constant);

    cmd.dispatch(gx, gy, 1);

    if (i > 1) {
      cmd.pipelineBarrier2(vk::DependencyInfo{}.setImageMemoryBarriers(barrier));
    }
    use_jfa_2 = !use_jfa_2;
  }
}

void Renderer2D::sdf_pass(vk::CommandBuffer cmd)
{
  cmd.bindPipeline(vk::PipelineBindPoint::eCompute, sdf_pipeline_.get());
  cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, sdf_pipeline_layout_.get(), 0, 1, &bindless_descriptor_set_,
                         0, nullptr);

  constexpr SDFPushConstant push_constant{
      .jfa_id = JFAImage,
      .sdf_id = SDFImage,
  };

  cmd.pushConstants(sdf_pipeline_layout_.get(), vk::ShaderStageFlagBits::eCompute, 0, sizeof(SDFPushConstant),
                    &push_constant);

  const uint32_t gx = (config_.scene_size.width + 7) / 8;
  const uint32_t gy = (config_.scene_size.height + 7) / 8;
  cmd.dispatch(gx, gy, 1);
}

void Renderer2D::cascades_pass(vk::CommandBuffer cmd)
{
  cmd.bindPipeline(vk::PipelineBindPoint::eCompute, cascades_pipeline_.get());
  cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, cascades_pipeline_layout_.get(), 0, 1,
                         &bindless_descriptor_set_, 0, nullptr);

  CascadesPushConstant push_constant{.sdf_id = SDFImage,
                                     .cascades_id = CascadesImage,
                                     .cascade_width = cascades_width_,
                                     .cascade_height = cascades_height_,
                                     .cascade = 0,
                                     .base_probe_size = base_probe_size_,
                                     .base_spacing = base_probe_spacing_,
                                     .base_probe_dir_count = base_probe_size_ * base_probe_size_,
                                     .base_length = base_probe_length_};

  const uint32_t gx = (cascades_width_ + 7) / 8;
  const uint32_t gy = (cascades_height_ + 7) / 8;

  for (uint32_t i = 0; i < config_.cascades.cascades; i++) {
    push_constant.cascade = i;

    cmd.pushConstants(cascades_pipeline_layout_.get(), vk::ShaderStageFlagBits::eCompute, 0,
                      sizeof(CascadesPushConstant), &push_constant);


    cmd.dispatch(gx, gy, 1);
  }
}

void Renderer2D::merge_pass(vk::CommandBuffer cmd)
{
  cmd.bindPipeline(vk::PipelineBindPoint::eCompute, merge_pipeline_.get());
  cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, merge_pipeline_layout_.get(), 0, 1, &bindless_descriptor_set_,
                         0, nullptr);

  const uint32_t gx = (cascades_width_ + 7) / 8;
  const uint32_t gy = (cascades_height_ + 7) / 8;

  MergePushConstant push_constant{
      .cascade_id = CascadesImage,
      .cascade = 0,
      .cascade_width = cascades_width_,
      .cascade_height = cascades_height_,
      .base_probe_size = base_probe_size_,
  };

  auto barrier =
      vk::ImageMemoryBarrier2{}
          .setSrcStageMask(vk::PipelineStageFlagBits2::eComputeShader)
          .setSrcAccessMask(vk::AccessFlagBits2::eShaderStorageWrite)
          .setDstStageMask(vk::PipelineStageFlagBits2::eComputeShader)
          .setDstAccessMask(vk::AccessFlagBits2::eShaderStorageWrite | vk::AccessFlagBits2::eShaderStorageRead)
          .setOldLayout(vk::ImageLayout::eGeneral)
          .setNewLayout(vk::ImageLayout::eGeneral)
          .setSrcQueueFamilyIndex(vk::QueueFamilyIgnored)
          .setDstQueueFamilyIndex(vk::QueueFamilyIgnored)
          .setSubresourceRange({vk::ImageAspectFlagBits::eColor, 0, 1, 0, vk::RemainingArrayLayers})
          .setImage(cascades_image_->image());

  for (uint32_t i = config_.cascades.cascades - 1; i-- > 0;) {
    push_constant.cascade = i;
    cmd.pushConstants(merge_pipeline_layout_.get(), vk::ShaderStageFlagBits::eCompute, 0, sizeof(MergePushConstant),
                      &push_constant);

    cmd.dispatch(gx, gy, 1);
    cmd.pipelineBarrier2(vk::DependencyInfo{}.setImageMemoryBarriers(barrier));
  }
}

void Renderer2D::light_pass(vk::CommandBuffer cmd)
{
  cmd.bindPipeline(vk::PipelineBindPoint::eCompute, light_pipeline_.get());
  cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, light_pipeline_layout_.get(), 0, 1, &bindless_descriptor_set_,
                         0, nullptr);

  const LightPushConstant push_constant{
      .cascade_id = CascadesImage,
      .light_id = LightImage,
      .sdf_id = SDFImage,
      .spacing = base_probe_spacing_,
      .probe_size = base_probe_size_,
  };

  cmd.pushConstants(light_pipeline_layout_.get(), vk::ShaderStageFlagBits::eCompute, 0, sizeof(LightPushConstant),
                    &push_constant);

  const uint32_t gx = (config_.scene_size.width + 7) / 8;
  const uint32_t gy = (config_.scene_size.height + 7) / 8;
  cmd.dispatch(gx, gy, 1);
}

void Renderer2D::pull_push_pass(vk::CommandBuffer cmd)
{
  cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pull_pipeline_.get());
  cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pull_push_pipeline_layout_.get(), 0, 1,
                         &bindless_descriptor_set_, 0, nullptr);

  auto barrier = vk::ImageMemoryBarrier2{}
                     .setSrcStageMask(vk::PipelineStageFlagBits2::eComputeShader)
                     .setSrcAccessMask(vk::AccessFlagBits2::eShaderStorageWrite)
                     .setDstStageMask(vk::PipelineStageFlagBits2::eComputeShader)
                     .setDstAccessMask(vk::AccessFlagBits2::eShaderStorageRead)
                     .setOldLayout(vk::ImageLayout::eGeneral)
                     .setNewLayout(vk::ImageLayout::eGeneral)
                     .setSrcQueueFamilyIndex(vk::QueueFamilyIgnored)
                     .setDstQueueFamilyIndex(vk::QueueFamilyIgnored)
                     .setSubresourceRange({vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1})
                     .setImage(light_image_->image());

  PullPushConstant push_constant{.read_id = 0, .write_id = 0, .width = 0, .height = 0};

  for (uint32_t i = 1; i < light_mips_; i++) {
    push_constant.read_id = LightImage + i - 1;
    push_constant.write_id = LightImage + i;
    push_constant.width = config_.scene_size.width >> i;
    push_constant.height = config_.scene_size.height >> i;

    cmd.pushConstants(pull_push_pipeline_layout_.get(), vk::ShaderStageFlagBits::eCompute, 0, sizeof(PullPushConstant),
                      &push_constant);

    if (i > 1) {
      barrier.subresourceRange.setBaseMipLevel(i - 1);
      cmd.pipelineBarrier2(vk::DependencyInfo{}.setImageMemoryBarriers(barrier));
    }

    const uint32_t gx = (push_constant.width + 7) / 8;
    const uint32_t gy = (push_constant.height + 7) / 8;
    cmd.dispatch(gx, gy, 1);
  }

  cmd.bindPipeline(vk::PipelineBindPoint::eCompute, push_pipeline_.get());

  std::array barriers = {barrier, barrier};
  barriers[0].subresourceRange.baseMipLevel += 1;
  barriers[1].setSrcAccessMask(vk::AccessFlagBits2::eShaderStorageRead);
  barriers[1].setDstAccessMask(vk::AccessFlagBits2::eShaderStorageWrite);
  cmd.pipelineBarrier2(vk::DependencyInfo{}.setImageMemoryBarriers(barriers));

  for (uint32_t i = light_mips_; i-- > 1;) {
    push_constant.read_id = LightImage + i;
    push_constant.write_id = LightImage + i - 1;
    push_constant.width = config_.scene_size.width >> (i - 1);
    push_constant.height = config_.scene_size.height >> (i - 1);

    cmd.pushConstants(pull_push_pipeline_layout_.get(), vk::ShaderStageFlagBits::eCompute, 0, sizeof(PullPushConstant),
                      &push_constant);

    if (i < 9) {
      barrier.subresourceRange.setBaseMipLevel(i);
      cmd.pipelineBarrier2(vk::DependencyInfo{}.setImageMemoryBarriers(barrier));
    }

    const uint32_t gx = (push_constant.width + 7) / 8;
    const uint32_t gy = (push_constant.height + 7) / 8;
    cmd.dispatch(gx, gy, 1);
  }
}

void Renderer2D::composite_pass(vk::CommandBuffer cmd)
{
  cmd.bindPipeline(vk::PipelineBindPoint::eCompute, composite_pipeline_.get());
  cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, composite_pipeline_layout_.get(), 0, 1,
                         &bindless_descriptor_set_, 0, nullptr);

  const CompositePushConstant push_constant{.composite_id = CompositeImage,
                                            .light_id = LightImage,
                                            .sdf_id = SDFImage,
                                            .scene_id = SceneImage,
                                            .rim = config_.composite.rim};

  cmd.pushConstants(composite_pipeline_layout_.get(), vk::ShaderStageFlagBits::eCompute, 0,
                    sizeof(CompositePushConstant), &push_constant);

  const uint32_t gx = (config_.scene_size.width + 7) / 8;
  const uint32_t gy = (config_.scene_size.height + 7) / 8;
  cmd.dispatch(gx, gy, 1);
}

void Renderer2D::blit_pass(vk::CommandBuffer cmd)
{
  cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, blit_pipeline_.get());
  cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, blit_pipeline_layout_.get(), 0, 1, &bindless_descriptor_set_,
                         0, nullptr);

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

  constexpr BlitPushConstant push_constant{
      .composite_id = CompositeTexture,
  };

  cmd.pushConstants(blit_pipeline_layout_.get(), vk::ShaderStageFlagBits::eFragment, 0, sizeof(BlitPushConstant),
                    &push_constant);

  cmd.draw(3, 1, 0, 0);
}

void Renderer2D::imgui_pass(vk::CommandBuffer cmd)
{
  ImGui_ImplVulkan_NewFrame();
  ImGui_ImplSDL3_NewFrame();
  ImGui::NewFrame();

  ImGui::Begin("Config");

  constexpr uint32_t min_draw_size = 1;
  constexpr uint32_t max_draw_size = 100;
  ImGui::SliderScalar("Draw size", ImGuiDataType_U32, &config_.drawing.size, &min_draw_size, &max_draw_size);

  ImGui::Text("Scene size");
  ImGui::SameLine();
  ImGui::PushItemWidth(80.0);
  ImGui::InputScalar("W", ImGuiDataType_U32, &config_.scene_size.width);
  ImGui::SameLine();
  ImGui::InputScalar("H", ImGuiDataType_U32, &config_.scene_size.height);
  ImGui::PopItemWidth();

  constexpr uint32_t min_cascades = 1;
  constexpr uint32_t max_cascades = 10;
  ImGui::SliderScalar("Cascades", ImGuiDataType_U32, &config_.cascades.cascades, &min_cascades, &max_cascades);

  ImGui::InputFloat("Base Spacing", &config_.cascades.base_spacing);
  ImGui::InputFloat("Base Interval", &config_.cascades.base_interval);
  ImGui::InputFloat("Base Length", &config_.cascades.base_length);

  for (size_t i = 0; i < materials_.size(); i++) {
    const Material& material = materials_.at(i);
    const ImVec4 color{material.color[0], material.color[1], material.color[2], 1.0f};
    const ImVec4 radiance{material.radiance[0], material.radiance[1], material.radiance[2], 1.0f};
    const bool has_radiance = !(radiance.x == 0.0f && radiance.y == 0.0f && radiance.z == 0.0f);
    if (i > 0) ImGui::SameLine();
    if (ImGui::ColorButton((std::string("mat") + std::to_string(i)).c_str(), has_radiance ? radiance : color,
                           ImGuiColorEditFlags_NoTooltip)) {
      draw_material_ = static_cast<uint8_t>(i);
    }

    if (has_radiance && ImGui::IsItemHovered()) {
      if (ImGui::BeginTooltip()) {
        ImGui::Text("Base color:");
        ImGui::SameLine();
        ImGui::ColorButton((std::string("mattooltip") + std::to_string(i)).c_str(), color,
                           ImGuiColorEditFlags_NoTooltip);
        ImGui::EndTooltip();
      }
    }
  }

  ImGui::SliderFloat("Rim", &config_.composite.rim, 0.0f, 100.0f);

  if (ImGui::Button("Recompile framework")) {
    should_compile_ = true;
  }
  ImGui::End();

  ImGui::Render();
  ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);
}

void Renderer2D::import_swapchain_images()
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

void Renderer2D::import_images()
{
  fwrk::ImageImportInfo image_import_info{
      .type = VK_IMAGE_TYPE_2D,
      .size = {.width = config_.scene_size.width, .height = config_.scene_size.height, .depth = 1},
      .format = VK_FORMAT_R8_UINT,
      .state = fwrk::PhysicalState::Undefined};

  if (scene_id_) {
    context_.update_image(scene_id_, image_import_info, scene_image_->image());
  } else {
    scene_id_ = context_.import_image(image_import_info, scene_image_->image());
  }

  image_import_info.format = VK_FORMAT_R32G32B32A32_SINT;
  if (jfa_id_) {
    context_.update_image(jfa_id_, image_import_info, jfa_image_->image());
  } else {
    jfa_id_ = context_.import_image(image_import_info, jfa_image_->image());
  }

  image_import_info.format = VK_FORMAT_R32_SFLOAT;
  if (sdf_id_) {
    context_.update_image(sdf_id_, image_import_info, sdf_image_->image());
  } else {
    sdf_id_ = context_.import_image(image_import_info, sdf_image_->image());
  }

  image_import_info.format = VK_FORMAT_R16G16B16A16_SFLOAT;
  if (light_id_) {
    context_.update_image(light_id_, image_import_info, light_image_->image());
  } else {
    light_id_ = context_.import_image(image_import_info, light_image_->image());
  }

  if (composite_id_) {
    context_.update_image(composite_id_, image_import_info, composite_image_->image());
  } else {
    composite_id_ = context_.import_image(image_import_info, composite_image_->image());
  }

  image_import_info.size.width = cascades_width_;
  image_import_info.size.height = cascades_height_;
  if (cascades_id_) {
    context_.update_image(cascades_id_, image_import_info, cascades_image_->image());
  } else {
    cascades_id_ = context_.import_image(image_import_info, cascades_image_->image());
  }
}

void Renderer2D::create_images()
{
  device_.get().waitIdle();

  scene_image_.emplace(device_.get_allocator(), ImageDesc{}
                                                    .set_extent(config_.scene_size.width, config_.scene_size.height)
                                                    .set_format(vk::Format::eR8Uint)
                                                    .set_usage(vk::ImageUsageFlagBits::eStorage));

  jfa_image_.emplace(device_.get_allocator(), ImageDesc{}
                                                  .set_extent(config_.scene_size.width, config_.scene_size.height)
                                                  .set_format(vk::Format::eR32G32B32A32Sint)
                                                  .set_layers(2)
                                                  .set_usage(vk::ImageUsageFlagBits::eStorage));

  sdf_image_.emplace(device_.get_allocator(), ImageDesc{}
                                                  .set_extent(config_.scene_size.width, config_.scene_size.height)
                                                  .set_format(vk::Format::eR32Sfloat)
                                                  .set_usage(vk::ImageUsageFlagBits::eStorage));

  light_mips_ = static_cast<uint32_t>(std::ceil(std::log2(config_.scene_size.width)));
  light_image_.emplace(device_.get_allocator(), ImageDesc{}
                                                    .set_extent(config_.scene_size.width, config_.scene_size.height)
                                                    .set_mips(light_mips_)
                                                    .set_format(vk::Format::eR16G16B16A16Sfloat)
                                                    .set_usage(vk::ImageUsageFlagBits::eStorage));

  composite_image_.emplace(device_.get_allocator(),
                           ImageDesc{}
                               .set_extent(config_.scene_size.width, config_.scene_size.height)
                               .set_format(vk::Format::eR16G16B16A16Sfloat)
                               .set_usage(vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled));


  scene_view_ = scene_image_->create_image_view(device_.get(), vk::ImageAspectFlagBits::eColor);
  jfa_view_ =
      jfa_image_->create_image_view(device_.get(), vk::ImageAspectFlagBits::eColor, vk::ImageViewType::e2DArray);
  sdf_view_ = sdf_image_->create_image_view(device_.get(), vk::ImageAspectFlagBits::eColor);
  composite_view_ = composite_image_->create_image_view(device_.get(), vk::ImageAspectFlagBits::eColor);

  light_views_.resize(light_mips_);
  for (uint32_t i = 0; i < light_mips_; i++) {
    light_views_[i] =
        light_image_->create_image_view(device_.get(), vk::ImageAspectFlagBits::eColor, vk::ImageViewType::e2D, i);
  }

  void* ubo_data;
  vmaMapMemory(device_.get_allocator(), ubo_->allocation(), &ubo_data);
  const UBOData ubo{.image_width = config_.scene_size.width, .image_height = config_.scene_size.height};
  memcpy(ubo_data, &ubo, sizeof(UBOData));
  vmaUnmapMemory(device_.get_allocator(), ubo_->allocation());
}

void Renderer2D::create_cascade_image()
{
  const auto base_probe_dir_count = static_cast<uint32_t>(std::round(360.0f / config_.cascades.base_interval));
  base_probe_size_ = static_cast<uint32_t>(std::ceil(std::sqrt(base_probe_dir_count)));
  const auto probe_per_row =
      static_cast<uint32_t>(std::floor(static_cast<float>(config_.scene_size.width) / config_.cascades.base_spacing));
  const auto probe_per_col =
      static_cast<uint32_t>(std::floor(static_cast<float>(config_.scene_size.height) / config_.cascades.base_spacing));
  cascades_width_ = probe_per_row * base_probe_size_;
  cascades_height_ = probe_per_col * base_probe_size_;

  base_probe_spacing_ = config_.cascades.base_spacing;
  base_probe_length_ = config_.cascades.base_length;

  cascades_image_.emplace(device_.get_allocator(), ImageDesc{}
                                                       .set_extent(cascades_width_, cascades_height_)
                                                       .set_format(vk::Format::eR16G16B16A16Sfloat)
                                                       .set_layers(config_.cascades.cascades)
                                                       .set_usage(vk::ImageUsageFlagBits::eStorage));
  cascades_view_ =
      cascades_image_->create_image_view(device_.get(), vk::ImageAspectFlagBits::eColor, vk::ImageViewType::e2DArray);
}

void Renderer2D::update_descriptor()
{
  DescriptorWriter writer{};
  writer.add_image(0, SceneImage, vk::DescriptorType::eStorageImage, scene_view_.get(), vk::ImageLayout::eGeneral)
      .add_image(0, JFAImage, vk::DescriptorType::eStorageImage, jfa_view_.get(), vk::ImageLayout::eGeneral)
      .add_image(0, SDFImage, vk::DescriptorType::eStorageImage, sdf_view_.get(), vk::ImageLayout::eGeneral)
      .add_image(0, CascadesImage, vk::DescriptorType::eStorageImage, cascades_view_.get(), vk::ImageLayout::eGeneral)
      .add_image(0, CompositeImage, vk::DescriptorType::eStorageImage, composite_view_.get(), vk::ImageLayout::eGeneral)
      .add_image(2, CompositeTexture, vk::DescriptorType::eSampledImage, composite_view_.get(),
                 vk::ImageLayout::eShaderReadOnlyOptimal);

  for (uint32_t i = 0; i < light_mips_; i++) {
    writer.add_image(0, LightImage + i, vk::DescriptorType::eStorageImage, light_views_[i].get(),
                     vk::ImageLayout::eGeneral);
  }

  writer.update(device_.get(), bindless_descriptor_set_);
}

VkSurfaceKHR Renderer2D::create_surface(const Window& window, const Instance& instance)
{
  VkSurfaceKHR surface = VK_NULL_HANDLE;
  if (!SDL_Vulkan_CreateSurface(window.get(), instance.get(), nullptr, &surface)) {
    throw std::runtime_error("Failed to create Vulkan Surface with SDL");
  }
  return surface;
}
