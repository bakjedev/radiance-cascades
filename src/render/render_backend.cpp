#include "render_backend.hpp"

#include "imgui_impl_vulkan.h"
#include "src/vulkan/descriptor.hpp"

RenderBackend::RenderBackend(const RendererCreateInfo& create_info) :
    instance_(create_info.instance_extensions), device_(instance_.get(), create_info.create_surface(instance_.get())),
    swapchain_(device_, create_info.initial_extent, vk::ImageUsageFlagBits::eColorAttachment),
    context_(create_fwrk_context()), immediate_(device_)
{
  for (Frame& frame: frames_) {
    create_frame(frame, device_);
  }

  submit_semaphores_.resize(swapchain_.image_count());
  constexpr vk::SemaphoreCreateInfo semaphore_create_info{};
  for (auto& semaphore: submit_semaphores_) {
    semaphore = device_.get().createSemaphoreUnique(semaphore_create_info);
  }

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
  descriptor_pool_ =
      create_descriptor_pool(device_.get(), sizes, 1, vk::DescriptorPoolCreateFlagBits::eUpdateAfterBind);

  // Bindless
  {
    bindless_descriptor_set_layout_ = create_descriptor_set_layout(
        device_.get(),
        DescriptorSetLayoutDesc{}
            .add_binding(0, vk::DescriptorType::eStorageImage, vk::ShaderStageFlagBits::eAll,
                         vk::DescriptorBindingFlagBits::ePartiallyBound, 100)
            .add_binding(
                1, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eAll,
                vk::DescriptorBindingFlagBits::ePartiallyBound | vk::DescriptorBindingFlagBits::eUpdateAfterBind, 100)
            .add_binding(2, vk::DescriptorType::eSampledImage, vk::ShaderStageFlagBits::eAll,
                         vk::DescriptorBindingFlagBits::ePartiallyBound, 100)
            .add_binding(3, vk::DescriptorType::eSampler, vk::ShaderStageFlagBits::eAll,
                         vk::DescriptorBindingFlagBits::ePartiallyBound, 100)
            .add_binding(4, vk::DescriptorType::eUniformBuffer, vk::ShaderStageFlagBits::eAll,
                         vk::DescriptorBindingFlagBits::ePartiallyBound, 100)
            .set_layout_flags(vk::DescriptorSetLayoutCreateFlagBits::eUpdateAfterBindPool));

    bindless_descriptor_set_ = device_.get()
                                   .allocateDescriptorSets(vk::DescriptorSetAllocateInfo{}
                                                               .setDescriptorPool(*descriptor_pool_)
                                                               .setSetLayouts(*bindless_descriptor_set_layout_))
                                   .front();
  }

  // ----------------------------------------
  // ImGui
  // ----------------------------------------
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

  auto imgui_shader = create_info.load_shader("imgui.slang.spv");
  imgui_vulkan_info.CustomShaderVertCreateInfo = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, nullptr, 0U,
                                                  imgui_shader.size(),
                                                  reinterpret_cast<const uint32_t*>(imgui_shader.data())};

  ImGui_ImplVulkan_Init(&imgui_vulkan_info);

  // ----------------------------------------
  // Imports
  // ----------------------------------------
  import_swapchain_images();
  swapchain_proxy_ = context_.create_proxy();
}

RenderBackend::~RenderBackend()
{
  device_.get().waitIdle();
  ImGui_ImplVulkan_Shutdown();
}

bool RenderBackend::begin_frame(const vk::Extent2D extent)
{
  if (extent.width == 0 || extent.height == 0) return false;
  const Frame& frame = frames_.at(current_frame_);

  vk::Result result = device_.get().waitForFences(frame.in_flight.get(), vk::True, UINT64_MAX);
  if (result != vk::Result::eSuccess) {
    throw std::runtime_error("Failed to wait for fences: " + vk::to_string(result));
  }

  if (needs_recreate_ || extent != swapchain_.extent()) {
    device_.get().waitIdle();
    swapchain_.recreate({extent.width, extent.height});
    import_swapchain_images();
    needs_recreate_ = false;
    should_compile_ = true;
  }

  const auto image_index = swapchain_.acquire_next_image(frame.image_available.get());
  if (!image_index) {
    needs_recreate_ = true;
    return false;
  }
  image_index_ = image_index.value();

  context_.update_proxy(swapchain_proxy_, swapchain_imports_.at(image_index_));

  device_.get().resetFences(frame.in_flight.get());

  device_.get().resetCommandPool(frame.command_pool.get());
  frame.command_buffer->begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
  return true;
}

void RenderBackend::execute_frame()
{
  context_.graph().execute(frames_.at(current_frame_).command_buffer.get(), current_frame_);
}

void RenderBackend::end_frame()
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
    needs_recreate_ = true;
  }

  current_frame_ = (current_frame_ + 1) % frames_in_flight;
}

void RenderBackend::window_resize(const int width, const int height)
{
  swapchain_.recreate({static_cast<uint32_t>(width), static_cast<uint32_t>(height)});
  import_swapchain_images();
}

fwrk::Context& RenderBackend::framework() { return context_; }

Device& RenderBackend::device() { return device_; }

ImmediateSubmit& RenderBackend::immediate() { return immediate_; }

const Frame& RenderBackend::get_current_frame() const { return frames_.at(current_frame_); }

vk::Format RenderBackend::swapchain_format() const { return swapchain_.format(); }

fwrk::ResourceID RenderBackend::swapchain_proxy() const { return swapchain_proxy_; }

const vk::DescriptorSetLayout& RenderBackend::bindless_layout() const { return bindless_descriptor_set_layout_.get(); }

const vk::DescriptorSet& RenderBackend::bindless_set() const { return bindless_descriptor_set_; }

vk::Sampler RenderBackend::sampler() const { return sampler_.get(); }

fwrk::Graph::FrameTimings RenderBackend::get_timings() { return context_.graph().get_timings(); }

void RenderBackend::request_recompile() { should_compile_ = true; }

bool RenderBackend::should_compile() const { return should_compile_; }

void RenderBackend::imgui_new_frame() { ImGui_ImplVulkan_NewFrame(); }

uint32_t RenderBackend::frame_index() const { return current_frame_; }

void RenderBackend::import_swapchain_images()
{
  swapchain_imports_.resize(swapchain_.image_count());

  const fwrk::ImageImportInfo swapchain_image_info{.type = VK_IMAGE_TYPE_2D,
                                                   .size = {swapchain_.extent().width, swapchain_.extent().height, 1},
                                                   .format = static_cast<VkFormat>(swapchain_.format()),
                                                   .mip_count = 1,
                                                   .layer_count = 1,
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

fwrk::Context RenderBackend::create_fwrk_context() const
{
  return fwrk::Context(
      {.vulkan = {instance_.get(), device_.get_physical(), device_.get(), frames_in_flight},
       .allocator = nullptr,
       .debug_markers =
           fwrk::DebugMarkerInfo{.cmd_begin = reinterpret_cast<PFN_vkCmdBeginDebugUtilsLabelEXT>(
                                     vkGetInstanceProcAddr(instance_.get(), "vkCmdBeginDebugUtilsLabelEXT")),
                                 .cmd_end = reinterpret_cast<PFN_vkCmdEndDebugUtilsLabelEXT>(
                                     vkGetInstanceProcAddr(instance_.get(), "vkCmdEndDebugUtilsLabelEXT"))},
       .timestamps = true});
}
