#include "renderer_2d.hpp"
#include <cmath>
#include "backend/descriptor.hpp"
#include "backend/pipeline.hpp"
#include "imgui_impl_vulkan.h"

namespace {
  struct DrawPushConstant {
    uint draw_id;
    uint32_t s;
    int32_t x;
    int32_t y;
    uint8_t material_id;
  };

  struct ConvertPushConstant {
    uint32_t draw_id;
  };

  struct SpritePushConstant {
    uint32_t sprite_textures_id;
    uint32_t sprite_buffer_id;
  };

  struct JFAPushConstant {
    uint32_t jfa_id;
    uint32_t read_idx;
    uint32_t k;
  };

  struct SDFPushConstant {
    uint32_t jfa_id;
    uint32_t sdf_id;
    uint32_t layer;
  };

  struct CascadesPushConstant {
    uint32_t sdf_id;
    uint32_t cascades_rw_id;
    uint32_t cascades_id;
    uint32_t scene_rad_id;
    uint32_t cascade_width;
    uint32_t cascade_height;
    uint32_t cascade_idx;
    uint32_t cascade_count;
    uint32_t base_probe_size;
    float base_spacing;
    uint32_t base_probe_dir_count;
    float base_length;
    uint32_t light_id;
    uint32_t scene_col_id;
    bool first;
  };

  struct UBOData {
    uint32_t image_width;
    uint32_t image_height;
  };

  struct LightPushConstant {
    uint32_t cascade_id;
    uint32_t light_id;
    uint32_t sdf_id;
    float spacing;
    uint32_t probe_size;
  };

  struct BlurPushConstant {
    uint32_t light_id;
    uint32_t blur_id;
    uint32_t blur_rw_id;
    uint32_t sampler_id;
    uint32_t pass;
    uint32_t sample_count;
    uint32_t weights_id;
    uint32_t offsets_id;
  };

  struct BlurResolvePushConstant {
    uint32_t light_rw_id;
    uint32_t blur_id;
    uint32_t sampler_id;
  };

  struct CompositePushConstant {
    uint32_t composite_id;
    uint32_t light_id;
    uint32_t sdf_id;
    uint scene_col_id;
    uint scene_rad_id;
    float rim;
  };

  struct BlitPushConstant {
    uint32_t composite_id;
    bool aces;
  };

  enum ImageDescriptors {
    DrawImage,
    JFAImage,
    SDFImage,
    CascadesImage,
    LightImage,
    BlurImage,
    CompositeImage,
  };

  enum TextureDescriptors {
    DrawTexture = 0,
    SceneColorTexture,
    SceneRadianceTexture,
    LightTexture,
    BlurTexture,
    BlurResolveTexture,
    CompositeTexture,
    CascadesTexture,
    SpriteTextures,
  };

  enum BufferDescriptors {
    MaterialBuffer = 0,
    GaussianWeightsBuffer,
    GaussianOffsetsBuffer,
    SpriteBuffer,
  };

  enum SamplerDescriptors {
    DefaultSampler = 0,
    LinearSampler,
  };
} // namespace

Renderer2D::Renderer2D(const RendererCreateInfo& create_info) :
    instance_(create_info.instance_extensions), device_(instance_.get(), create_info.create_surface(instance_.get())),
    swapchain_(device_, create_info.initial_extent, vk::ImageUsageFlagBits::eColorAttachment),
    context_(create_fwrk_context()), immediate_(device_)
{
  for (Frame& frame: frames_) {
    create_frame(frame, device_, sizeof(SpriteGPU) * max_sprites);
  }

  submit_semaphores_.resize(swapchain_.image_count());
  constexpr vk::SemaphoreCreateInfo semaphore_create_info{};
  for (auto& semaphore: submit_semaphores_) {
    semaphore = device_.get().createSemaphoreUnique(semaphore_create_info);
  }

  // ----------------------------------------
  // Buffers
  // ----------------------------------------
  material_buffer_.emplace(device_.get_allocator(), device_.get(),
                           BufferDesc{.alloc_flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT}
                               .set_size(sizeof(Material) * 256)
                               .set_usage(vk::BufferUsageFlagBits::eStorageBuffer)
                               .set_debug_name("Material buffer"));

  materials_.push_back(Material{.color = {1.0f, 1.0f, 1.0f}, .radiance = {0.0f, 0.0f, 0.0f}});
  materials_.push_back(Material{.color = {1.0f, 1.0f, 1.0f}, .radiance = {0.0f, 0.0f, 0.0f}});
  materials_.push_back(Material{.color = {0.58f, 0.63f, 0.57f}, .radiance = {0.0f, 0.0f, 0.0f}});
  materials_.push_back(Material{.color = {1.0f, 1.0f, 1.0f}, .radiance = {1.0f, 1.0f, 1.0f}});
  materials_.push_back(Material{.color = {0.29f, 0.95f, 0.19f}, .radiance = {0.58f, 1.90f, 0.38f}});
  materials_.push_back(Material{.color = {0.94f, 0.21f, 0.28f}, .radiance = {0.94f, 0.21f, 0.28f}});
  materials_.push_back(Material{.color = {0.31f, 0.48f, 0.88f}, .radiance = {0.31f, 0.48f, 0.88f}});

  void* material_data;
  vmaMapMemory(device_.get_allocator(), material_buffer_->allocation(), &material_data);
  memcpy(material_data, materials_.data(), sizeof(Material) * materials_.size());
  vmaUnmapMemory(device_.get_allocator(), material_buffer_->allocation());


  ubo_.emplace(device_.get_allocator(), device_.get(),
               BufferDesc{}
                   .set_size(sizeof(UBOData))
                   .set_usage(vk::BufferUsageFlagBits::eUniformBuffer)
                   .set_allocate_flags(VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT)
                   .set_debug_name("UBO"));

  gaussian_weights_buffer_.emplace(
      device_.get_allocator(), device_.get(),
      BufferDesc{}
          .set_size(512 * sizeof(float))
          .set_usage(vk::BufferUsageFlagBits::eStorageBuffer)
          .set_allocate_flags(VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT)
          .set_debug_name("Gaussian Weights"));

  gaussian_offsets_buffer_.emplace(
      device_.get_allocator(), device_.get(),
      BufferDesc{}
          .set_size(512 * sizeof(float))
          .set_usage(vk::BufferUsageFlagBits::eStorageBuffer)
          .set_allocate_flags(VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT)
          .set_debug_name("Gaussian Offsets"));

  // ----------------------------------------
  // Sampler
  // ----------------------------------------
  vk::SamplerCreateInfo sampler_info{};
  sampler_ = device_.get().createSamplerUnique(sampler_info);
  sampler_info.setAddressModeU(vk::SamplerAddressMode::eClampToBorder);
  sampler_info.setAddressModeV(vk::SamplerAddressMode::eClampToBorder);
  sampler_info.setMinFilter(vk::Filter::eLinear);
  sampler_info.setMagFilter(vk::Filter::eLinear);
  cascades_sampler_ = device_.get().createSamplerUnique(sampler_info);

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

  DescriptorWriter{}
      .add_buffer(1, MaterialBuffer, vk::DescriptorType::eStorageBuffer, material_buffer_->buffer())
      .add_buffer(1, GaussianWeightsBuffer, vk::DescriptorType::eStorageBuffer, gaussian_weights_buffer_->buffer())
      .add_buffer(1, GaussianOffsetsBuffer, vk::DescriptorType::eStorageBuffer, gaussian_offsets_buffer_->buffer())
      .add_image(3, DefaultSampler, vk::DescriptorType::eSampler, nullptr, vk::ImageLayout::eUndefined, sampler_.get())
      .add_image(3, LinearSampler, vk::DescriptorType::eSampler, nullptr, vk::ImageLayout::eUndefined,
                 cascades_sampler_.get())
      .add_buffer(4, 0, vk::DescriptorType::eUniformBuffer, ubo_->buffer())
      .update(device_.get(), bindless_descriptor_set_);

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
  // Pipelines
  // ----------------------------------------

  std::vector<vk::DebugUtilsObjectNameInfoEXT> debug_object_names;
  debug_object_names.reserve(10);

  auto handle_to_uint64_t = []<typename T>(T handle) -> uint64_t {
    return reinterpret_cast<uint64_t>(static_cast<T::CType>(handle));
  };

  // Draw pipeline
  {
    vk::PushConstantRange draw_push{vk::ShaderStageFlagBits::eCompute, 0, sizeof(DrawPushConstant)};

    const auto draw_shader = create_info.load_shader("draw.slang.spv");
    draw_shader_module_ = create_shader_module(device_.get(), draw_shader);

    draw_pipeline_layout_ =
        create_pipeline_layout(device_.get(), {&bindless_descriptor_set_layout_.get(), 1}, {&draw_push, 1});

    const ComputePipelineDesc draw_pipeline_desc{.module = draw_shader_module_.get(),
                                                 .layout = draw_pipeline_layout_.get()};

    draw_pipeline_ = create_compute_pipeline(device_.get(), draw_pipeline_desc);

    debug_object_names.emplace_back(vk::ObjectType::ePipeline, handle_to_uint64_t(draw_pipeline_.get()),
                                    "Draw Pipeline");
  }

  // Convert pipeline
  {
    vk::PushConstantRange convert_push{vk::ShaderStageFlagBits::eFragment, 0, sizeof(ConvertPushConstant)};

    const auto convert_shader = create_info.load_shader("convert.slang.spv");
    convert_shader_module_ = create_shader_module(device_.get(), convert_shader);

    convert_pipeline_layout_ =
        create_pipeline_layout(device_.get(), {&bindless_descriptor_set_layout_.get(), 1}, {&convert_push, 1});

    GraphicsPipelineDesc convert_pipeline_desc{};
    convert_pipeline_desc.stages.emplace_back(vk::ShaderStageFlagBits::eVertex, convert_shader_module_.get(),
                                              "vs_main");
    convert_pipeline_desc.stages.emplace_back(vk::ShaderStageFlagBits::eFragment, convert_shader_module_.get(),
                                              "fs_main");

    convert_pipeline_desc.depth_stencil.depthTestEnable = vk::False;
    convert_pipeline_desc.depth_stencil.depthWriteEnable = vk::False;

    convert_pipeline_desc.add_attachment(vk::Format::eR8G8B8A8Srgb);
    convert_pipeline_desc.add_attachment(vk::Format::eB10G11R11UfloatPack32);
    convert_pipeline_desc.add_attachment(vk::Format::eR32G32B32A32Sint);

    convert_pipeline_desc.layout = convert_pipeline_layout_.get();

    convert_pipeline_ = create_graphics_pipeline(device_.get(), convert_pipeline_desc);

    debug_object_names.emplace_back(vk::ObjectType::ePipeline, handle_to_uint64_t(convert_pipeline_.get()),
                                    "Convert Pipeline");
  }

  // Sprite pipeline
  {
    vk::PushConstantRange sprite_push{vk::ShaderStageFlagBits::eVertex, 0, sizeof(SpritePushConstant)};

    const auto sprite_shader = create_info.load_shader("sprite.slang.spv");
    sprite_shader_module_ = create_shader_module(device_.get(), sprite_shader);

    sprite_pipeline_layout_ =
        create_pipeline_layout(device_.get(), {&bindless_descriptor_set_layout_.get(), 1}, {&sprite_push, 1});

    GraphicsPipelineDesc sprite_pipeline_desc{};
    sprite_pipeline_desc.stages.emplace_back(vk::ShaderStageFlagBits::eVertex, sprite_shader_module_.get(), "vs_main");
    sprite_pipeline_desc.stages.emplace_back(vk::ShaderStageFlagBits::eFragment, sprite_shader_module_.get(),
                                             "fs_main");

    sprite_pipeline_desc.input_assembly.setTopology(vk::PrimitiveTopology::eTriangleStrip);
    sprite_pipeline_desc.rasterization.cullMode = vk::CullModeFlagBits::eNone;
    sprite_pipeline_desc.depth_stencil.depthTestEnable = vk::False;
    sprite_pipeline_desc.depth_stencil.depthWriteEnable = vk::False;

    sprite_pipeline_desc.add_attachment(vk::Format::eR8G8B8A8Srgb);
    sprite_pipeline_desc.add_attachment(vk::Format::eB10G11R11UfloatPack32);
    sprite_pipeline_desc.add_attachment(vk::Format::eR32G32B32A32Sint);

    sprite_pipeline_desc.layout = sprite_pipeline_layout_.get();

    sprite_pipeline_ = create_graphics_pipeline(device_.get(), sprite_pipeline_desc);

    debug_object_names.emplace_back(vk::ObjectType::ePipeline, handle_to_uint64_t(sprite_pipeline_.get()),
                                    "Sprite Pipeline");
  }

  // JFA pipeline
  {
    vk::PushConstantRange jfa_push{vk::ShaderStageFlagBits::eCompute, 0, sizeof(JFAPushConstant)};

    const auto jfa_shader = create_info.load_shader("jfa.slang.spv");
    jfa_shader_module_ = create_shader_module(device_.get(), jfa_shader);

    jfa_pipeline_layout_ =
        create_pipeline_layout(device_.get(), {&bindless_descriptor_set_layout_.get(), 1}, {&jfa_push, 1});

    const ComputePipelineDesc jfa_pipeline_desc{.module = jfa_shader_module_.get(),
                                                .layout = jfa_pipeline_layout_.get()};

    jfa_pipeline_ = create_compute_pipeline(device_.get(), jfa_pipeline_desc);

    debug_object_names.emplace_back(vk::ObjectType::ePipeline, handle_to_uint64_t(jfa_pipeline_.get()), "JFA Pipeline");
  }

  // SDF pipeline
  {
    vk::PushConstantRange sdf_push{vk::ShaderStageFlagBits::eCompute, 0, sizeof(SDFPushConstant)};

    const auto sdf_shader = create_info.load_shader("sdf.slang.spv");
    sdf_shader_module_ = create_shader_module(device_.get(), sdf_shader);

    sdf_pipeline_layout_ =
        create_pipeline_layout(device_.get(), {&bindless_descriptor_set_layout_.get(), 1}, {&sdf_push, 1});

    const ComputePipelineDesc sdf_pipeline_desc{.module = sdf_shader_module_.get(),
                                                .layout = sdf_pipeline_layout_.get()};

    sdf_pipeline_ = create_compute_pipeline(device_.get(), sdf_pipeline_desc);

    debug_object_names.emplace_back(vk::ObjectType::ePipeline, handle_to_uint64_t(sdf_pipeline_.get()), "SDF Pipeline");
  }

  // Cascades pipeline
  {
    vk::PushConstantRange cascades_push{vk::ShaderStageFlagBits::eCompute, 0, sizeof(CascadesPushConstant)};

    const auto cascades_shader = create_info.load_shader("cascades.slang.spv");
    cascades_shader_module_ = create_shader_module(device_.get(), cascades_shader);

    cascades_pipeline_layout_ =
        create_pipeline_layout(device_.get(), {&bindless_descriptor_set_layout_.get(), 1}, {&cascades_push, 1});

    const ComputePipelineDesc cascades_pipeline_desc{.module = cascades_shader_module_.get(),
                                                     .layout = cascades_pipeline_layout_.get()};

    cascades_pipeline_ = create_compute_pipeline(device_.get(), cascades_pipeline_desc);

    debug_object_names.emplace_back(vk::ObjectType::ePipeline, handle_to_uint64_t(cascades_pipeline_.get()),
                                    "Cascades Pipeline");
  }

  // Light pipeline
  {
    vk::PushConstantRange light_push{vk::ShaderStageFlagBits::eCompute, 0, sizeof(LightPushConstant)};

    const auto light_shader = create_info.load_shader("light.slang.spv");
    light_shader_module_ = create_shader_module(device_.get(), light_shader);

    light_pipeline_layout_ =
        create_pipeline_layout(device_.get(), {&bindless_descriptor_set_layout_.get(), 1}, {&light_push, 1});

    const ComputePipelineDesc light_pipeline_desc{.module = light_shader_module_.get(),
                                                  .layout = light_pipeline_layout_.get()};

    light_pipeline_ = create_compute_pipeline(device_.get(), light_pipeline_desc);

    debug_object_names.emplace_back(vk::ObjectType::ePipeline, handle_to_uint64_t(light_pipeline_.get()),
                                    "Light Pipeline");
  }

  // Blur pipelines
  {
    vk::PushConstantRange blur_push{vk::ShaderStageFlagBits::eCompute, 0, sizeof(BlurPushConstant)};
    vk::PushConstantRange blur_resolve_push{vk::ShaderStageFlagBits::eCompute, 0, sizeof(BlurResolvePushConstant)};

    const auto blur_shader = create_info.load_shader("blur.slang.spv");
    blur_shader_module_ = create_shader_module(device_.get(), blur_shader);

    blur_pipeline_layout_ =
        create_pipeline_layout(device_.get(), {&bindless_descriptor_set_layout_.get(), 1}, {&blur_push, 1});

    const ComputePipelineDesc blur_pipeline_desc{
        .module = blur_shader_module_.get(),
        .entry = "blur",
        .layout = blur_pipeline_layout_.get(),
    };

    blur_pipeline_ = create_compute_pipeline(device_.get(), blur_pipeline_desc);

    debug_object_names.emplace_back(vk::ObjectType::ePipeline, handle_to_uint64_t(blur_pipeline_.get()),
                                    "Blur Pipeline");
    // resolve
    blur_resolve_pipeline_layout_ =
        create_pipeline_layout(device_.get(), {&bindless_descriptor_set_layout_.get(), 1}, {&blur_resolve_push, 1});

    const ComputePipelineDesc blur_resolve_pipeline_desc{
        .module = blur_shader_module_.get(),
        .entry = "resolve",
        .layout = blur_resolve_pipeline_layout_.get(),
    };

    blur_resolve_pipeline_ = create_compute_pipeline(device_.get(), blur_resolve_pipeline_desc);

    debug_object_names.emplace_back(vk::ObjectType::ePipeline, handle_to_uint64_t(blur_resolve_pipeline_.get()),
                                    "Blur Resolve Pipeline");
  }

  // Composite pipeline
  {
    vk::PushConstantRange composite_push{vk::ShaderStageFlagBits::eCompute, 0, sizeof(CompositePushConstant)};

    const auto composite_shader = create_info.load_shader("composite.slang.spv");
    composite_shader_module_ = create_shader_module(device_.get(), composite_shader);

    composite_pipeline_layout_ =
        create_pipeline_layout(device_.get(), {&bindless_descriptor_set_layout_.get(), 1}, {&composite_push, 1});

    const ComputePipelineDesc composite_pipeline_desc{.module = composite_shader_module_.get(),
                                                      .layout = composite_pipeline_layout_.get()};

    composite_pipeline_ = create_compute_pipeline(device_.get(), composite_pipeline_desc);

    debug_object_names.emplace_back(vk::ObjectType::ePipeline, handle_to_uint64_t(composite_pipeline_.get()),
                                    "Composite Pipeline");
  }

  // Blit pipeline
  {
    vk::PushConstantRange blit_push{vk::ShaderStageFlagBits::eFragment, 0, sizeof(BlitPushConstant)};

    const auto blit_shader = create_info.load_shader("blit.slang.spv");
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
}

void Renderer2D::render(RenderInfo info)
{
  info_ = std::move(info);
  if (begin_frame()) {
    run_frame();
    end_frame();
  }
}
void Renderer2D::window_resize(const int width, const int height)
{
  swapchain_.recreate({static_cast<uint32_t>(width), static_cast<uint32_t>(height)});
  import_swapchain_images();
}

Renderer2D::Texture Renderer2D::create_texture(const std::span<const uint8_t> data,
                                               const std::pair<uint32_t, uint32_t>& size)
{
  auto image = Image(device_.get_allocator(), device_.get(),
                     ImageDesc{}
                         .set_extent(size.first, size.second)
                         .set_format(vk::Format::eR8G8B8A8Srgb)
                         .set_usage(vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst)
                         .set_debug_name(std::format("Texture {}", texture_images_.size())));

  const vk::DeviceSize buffer_size = data.size() * sizeof(data[0]);
  const auto staging = Buffer(device_.get_allocator(), device_.get(),
                              BufferDesc{}
                                  .set_size(buffer_size)
                                  .set_usage(vk::BufferUsageFlagBits::eTransferSrc)
                                  .set_allocate_flags(VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT));

  void* mapped;
  vmaMapMemory(device_.get_allocator(), staging.allocation(), &mapped);
  memcpy(mapped, data.data(), buffer_size);
  vmaUnmapMemory(device_.get_allocator(), staging.allocation());

  immediate_.submit([&](vk::CommandBuffer cmd) {
    auto barrier = vk::ImageMemoryBarrier2{}
                       .setSrcStageMask(vk::PipelineStageFlagBits2::eNone)
                       .setSrcAccessMask(vk::AccessFlagBits2::eNone)
                       .setDstStageMask(vk::PipelineStageFlagBits2::eCopy)
                       .setDstAccessMask(vk::AccessFlagBits2::eTransferWrite)
                       .setOldLayout(vk::ImageLayout::eUndefined)
                       .setNewLayout(vk::ImageLayout::eTransferDstOptimal)
                       .setSrcQueueFamilyIndex(vk::QueueFamilyIgnored)
                       .setDstQueueFamilyIndex(vk::QueueFamilyIgnored)
                       .setSubresourceRange({vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1})
                       .setImage(image.image());
    cmd.pipelineBarrier2(vk::DependencyInfo{}.setImageMemoryBarriers(barrier));

    const vk::BufferImageCopy2 region{
        0, 0, 0, {vk::ImageAspectFlagBits::eColor, 0, 0, 1}, {0, 0, 0}, {size.first, size.second, 1}};

    cmd.copyBufferToImage2(
        vk::CopyBufferToImageInfo2{staging.buffer(), image.image(), vk::ImageLayout::eTransferDstOptimal, 1, &region});

    barrier.setSrcStageMask(vk::PipelineStageFlagBits2::eCopy);
    barrier.setSrcAccessMask(vk::AccessFlagBits2::eTransferWrite);
    barrier.setDstStageMask(vk::PipelineStageFlagBits2::eFragmentShader);
    barrier.setDstAccessMask(vk::AccessFlagBits2::eShaderSampledRead);
    barrier.setOldLayout(vk::ImageLayout::eTransferDstOptimal);
    barrier.setNewLayout(vk::ImageLayout::eShaderReadOnlyOptimal);
    cmd.pipelineBarrier2(vk::DependencyInfo{}.setImageMemoryBarriers(barrier));
  });

  texture_images_.emplace_back(std::move(image),
                               image.create_image_view(device_.get(), vk::ImageAspectFlagBits::eColor));
  return Texture{static_cast<uint32_t>(texture_images_.size() - 1)};
}

void Renderer2D::plot() { should_draw_ = true; }

void Renderer2D::draw_sprite(Sprite sprite) { sprites_.emplace_back(sprite); }

void Renderer2D::imgui_new_frame() { ImGui_ImplVulkan_NewFrame(); }

void Renderer2D::request_recompile() { should_compile_ = true; }

fwrk::Graph::FrameTimings Renderer2D::get_timings() { return context_.graph().get_timings(); }

bool Renderer2D::begin_frame()
{
  const Frame& frame = frames_.at(current_frame_);

  vk::Result result = device_.get().waitForFences(frame.in_flight.get(), vk::True, UINT64_MAX);
  if (result != vk::Result::eSuccess) {
    throw std::runtime_error("Failed to wait for fences: " + vk::to_string(result));
  }

  const auto image_index = swapchain_.acquire_next_image(frame.image_available.get());
  if (!image_index) {
    window_resize(info_.window_size.first, info_.window_size.second);
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

  upload_sprites();

  if (should_draw_ != did_draw_) {
    should_compile_ = true;
  }

  if (config_.composite.blur_radius != previous_gaussian_radius_) {
    calculate_gaussian(config_.composite.blur_radius, 3.0f);
    std::memcpy(gaussian_weights_buffer_->get_mapped_data(), gaussian_weights_.data(),
                gaussian_weights_.size() * sizeof(gaussian_weights_[0]));
    std::memcpy(gaussian_offsets_buffer_->get_mapped_data(), gaussian_offsets_.data(),
                gaussian_offsets_.size() * sizeof(gaussian_offsets_[0]));
    previous_gaussian_radius_ = config_.composite.blur_radius;
  }

  if (should_compile_) {
    should_compile_ = false;
    compile();
  }

  context_.update_proxy(swapchain_proxy_, swapchain_imports_[image_index_]);

  graph.execute(frame.command_buffer.get(), current_frame_);

  sprites_.clear();

  did_draw_ = should_draw_;
  should_draw_ = false;
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
    window_resize(info_.window_size.first, info_.window_size.second);
  }

  current_frame_ = (current_frame_ + 1) % frames_in_flight;
}

void Renderer2D::compile()
{
  fwrk::Graph& graph = context_.graph();

  if (!draw_image_ || draw_image_.value().extent().width != config_.scene_size.width) {
    create_images();
    create_cascade_image();
    import_images();
    update_descriptor();
  }

  // ------------------
  // Passes
  // ------------------
  if (should_draw_) {
    graph.add_compute_pass("Draw")
        .set_storage_image_write({.resource = {.id = draw_id_}})
        .set_execute([this](vk::CommandBuffer cmd) { draw_pass(cmd); });
  }

  graph.add_graphics_pass("Convert")
      .set_image_read({.resource = {.id = draw_id_}})
      .set_color_attachment(
          {.resource = {.id = scene_color_id_}, .load_op = fwrk::LoadOp::Clear, .store_op = fwrk::StoreOp::Store})
      .set_color_attachment(
          {.resource = {.id = scene_radiance_id_}, .load_op = fwrk::LoadOp::Clear, .store_op = fwrk::StoreOp::Store})
      .set_color_attachment(
          {.resource = {.id = jfa_id_}, .load_op = fwrk::LoadOp::Clear, .store_op = fwrk::StoreOp::Store})
      .set_execute([this](vk::CommandBuffer cmd) { convert_pass(cmd); });

  graph.add_graphics_pass("Sprites")
      .set_color_attachment(
          {.resource = {.id = scene_color_id_}, .load_op = fwrk::LoadOp::Load, .store_op = fwrk::StoreOp::Store})
      .set_color_attachment(
          {.resource = {.id = scene_radiance_id_}, .load_op = fwrk::LoadOp::Load, .store_op = fwrk::StoreOp::Store})
      .set_color_attachment(
          {.resource = {.id = jfa_id_}, .load_op = fwrk::LoadOp::Load, .store_op = fwrk::StoreOp::Store})
      .set_execute([this](vk::CommandBuffer cmd) { sprite_pass(cmd); });

  graph.add_compute_pass("JFA")
      .set_storage_image_write({.resource = {.id = jfa_id_}})
      .set_execute([this](vk::CommandBuffer cmd) { jfa_pass(cmd); });

  graph.add_compute_pass("SDF")
      .set_storage_image_read({.resource = {.id = jfa_id_}})
      .set_storage_image_write({.resource = {.id = sdf_id_}})
      .set_execute([this](vk::CommandBuffer cmd) { sdf_pass(cmd); });

  for (uint32_t i = 0; i < config_.cascades.bounce; i++) {
    auto cascade_pass = graph.add_compute_pass("Cascades")
                            .set_storage_image_read({.resource = {.id = sdf_id_}})
                            .set_storage_image_read({.resource = {.id = scene_color_id_}})
                            .set_storage_image_read({.resource = {.id = scene_radiance_id_}})
                            .set_storage_image_write({.resource = {.id = cascades_id_}});

    if (i != 0) cascade_pass.set_storage_image_read({.resource = {.id = light_id_}});

    cascade_pass.set_execute([this, i](vk::CommandBuffer cmd) { cascades_pass(cmd, i == 0); });

    graph.add_compute_pass("Light")
        .set_storage_image_read({.resource = {.id = cascades_id_}})
        .set_storage_image_write({.resource = {.id = light_id_}, .level_count = 1})
        .set_execute([this](vk::CommandBuffer cmd) { light_pass(cmd); });
  }

  graph.add_compute_pass("Blur")
      .set_image_read({.resource = {.id = light_id_}})
      .set_storage_image_write({.resource = {.id = blur_id_}})
      .set_execute([this](vk::CommandBuffer cmd) { blur_pass(cmd); });

  graph.add_compute_pass("Composite")
      .set_storage_image_read({.resource = {.id = light_id_}})
      .set_storage_image_read({.resource = {.id = sdf_id_}})
      .set_storage_image_read({.resource = {.id = scene_color_id_}})
      .set_storage_image_read({.resource = {.id = scene_radiance_id_}})
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
  cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, draw_pipeline_layout_.get(), 0, 1, &bindless_descriptor_set_,
                         0, nullptr);
  cmd.bindPipeline(vk::PipelineBindPoint::eCompute, draw_pipeline_.get());

  const auto [draw_x, draw_y] = screen_to_pos(info_.mouse_pos);
  const DrawPushConstant push_constant{.draw_id = DrawImage,
                                       .s = config_.drawing.size,
                                       .x = draw_x,
                                       .y = draw_y,
                                       .material_id = config_.drawing.material};

  cmd.pushConstants(draw_pipeline_layout_.get(), vk::ShaderStageFlagBits::eCompute, 0, sizeof(DrawPushConstant),
                    &push_constant);

  const uint32_t gs = (config_.drawing.size + 7) / 8;
  cmd.dispatch(gs, gs, 1);
}

void Renderer2D::convert_pass(vk::CommandBuffer cmd)
{
  cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, convert_pipeline_.get());
  cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, convert_pipeline_layout_.get(), 0, 1,
                         &bindless_descriptor_set_, 0, nullptr);

  const vk::Viewport viewport{0.0f, 0.0f, static_cast<float>(config_.scene_size.width),
                              static_cast<float>(config_.scene_size.height)};
  cmd.setViewport(0, 1, &viewport);

  const vk::Rect2D scissor{vk::Offset2D{0, 0}, vk::Extent2D{config_.scene_size.width, config_.scene_size.height}};
  cmd.setScissor(0, 1, &scissor);

  constexpr ConvertPushConstant push_constant{.draw_id = DrawTexture};

  cmd.pushConstants(convert_pipeline_layout_.get(), vk::ShaderStageFlagBits::eFragment, 0, sizeof(ConvertPushConstant),
                    &push_constant);

  cmd.draw(3, 1, 0, 0);
}

void Renderer2D::sprite_pass(vk::CommandBuffer cmd)
{
  cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, sprite_pipeline_.get());
  cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, sprite_pipeline_layout_.get(), 0, 1,
                         &bindless_descriptor_set_, 0, nullptr);

  const vk::Viewport viewport{0.0f, 0.0f, static_cast<float>(config_.scene_size.width),
                              static_cast<float>(config_.scene_size.height)};
  cmd.setViewport(0, 1, &viewport);

  const vk::Rect2D scissor{vk::Offset2D{0, 0}, vk::Extent2D{config_.scene_size.width, config_.scene_size.height}};
  cmd.setScissor(0, 1, &scissor);

  constexpr SpritePushConstant push_constant{.sprite_textures_id = SpriteTextures, .sprite_buffer_id = SpriteBuffer};

  cmd.pushConstants(sprite_pipeline_layout_.get(), vk::ShaderStageFlagBits::eVertex, 0, sizeof(SpritePushConstant),
                    &push_constant);

  cmd.draw(4, static_cast<uint32_t>(sprites_.size()), 0, 0);
}

void Renderer2D::jfa_pass(vk::CommandBuffer cmd)
{
  cmd.bindPipeline(vk::PipelineBindPoint::eCompute, jfa_pipeline_.get());
  cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, jfa_pipeline_layout_.get(), 0, 1, &bindless_descriptor_set_,
                         0, nullptr);

  auto barrier =
      vk::ImageMemoryBarrier2{}
          .setSrcStageMask(vk::PipelineStageFlagBits2::eComputeShader)
          .setSrcAccessMask(vk::AccessFlagBits2::eShaderStorageWrite)
          .setDstStageMask(vk::PipelineStageFlagBits2::eComputeShader)
          .setDstAccessMask(vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eShaderStorageWrite)
          .setOldLayout(vk::ImageLayout::eGeneral)
          .setNewLayout(vk::ImageLayout::eGeneral)
          .setSrcQueueFamilyIndex(vk::QueueFamilyIgnored)
          .setDstQueueFamilyIndex(vk::QueueFamilyIgnored)
          .setSubresourceRange(
              {vk::ImageAspectFlagBits::eColor, 0, vk::RemainingMipLevels, 0, vk::RemainingArrayLayers})
          .setImage(jfa_image_->image());

  vk::DebugUtilsLabelEXT debug_label{{}, {1.0f, 1.0f, 1.0f, 1.0f}};

  const auto k_start = std::max(config_.scene_size.width, config_.scene_size.height);
  const auto k_count = k_start <= 1 ? 0 : static_cast<uint32_t>(std::bit_width(k_start - 1));

  const uint32_t gx = (config_.scene_size.width + 7) / 8;
  const uint32_t gy = (config_.scene_size.height + 7) / 8;
  uint32_t read = 0;
  for (uint32_t i = 1; i <= k_count; i++) {
    std::string debug_name = std::format("JFA Iter {}/{}, r: {}", i, k_count, read);
    debug_label.pLabelName = debug_name.c_str();
    cmd.beginDebugUtilsLabelEXT(*debug_label);

    const JFAPushConstant push_constant{.jfa_id = JFAImage, .read_idx = read, .k = k_start >> i};
    cmd.pushConstants(jfa_pipeline_layout_.get(), vk::ShaderStageFlagBits::eCompute, 0, sizeof(JFAPushConstant),
                      &push_constant);

    cmd.dispatch(gx, gy, 1);
    cmd.pipelineBarrier2(vk::DependencyInfo{}.setImageMemoryBarriers(barrier));

    read = 1 - read;
    cmd.endDebugUtilsLabelEXT();
  }
  final_jfa_ = read;
}

void Renderer2D::sdf_pass(vk::CommandBuffer cmd)
{
  cmd.bindPipeline(vk::PipelineBindPoint::eCompute, sdf_pipeline_.get());
  cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, sdf_pipeline_layout_.get(), 0, 1, &bindless_descriptor_set_,
                         0, nullptr);

  const SDFPushConstant push_constant{.jfa_id = JFAImage, .sdf_id = SDFImage, .layer = final_jfa_};

  cmd.pushConstants(sdf_pipeline_layout_.get(), vk::ShaderStageFlagBits::eCompute, 0, sizeof(SDFPushConstant),
                    &push_constant);

  const uint32_t gx = (config_.scene_size.width + 7) / 8;
  const uint32_t gy = (config_.scene_size.height + 7) / 8;
  cmd.dispatch(gx, gy, 1);
}

void Renderer2D::cascades_pass(vk::CommandBuffer cmd, const bool first)
{
  cmd.bindPipeline(vk::PipelineBindPoint::eCompute, cascades_pipeline_.get());
  cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, cascades_pipeline_layout_.get(), 0, 1,
                         &bindless_descriptor_set_, 0, nullptr);

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
          .setSubresourceRange(
              {vk::ImageAspectFlagBits::eColor, 0, vk::RemainingMipLevels, 0, vk::RemainingArrayLayers})
          .setImage(cascades_image_->image());

  vk::DebugUtilsLabelEXT debug_label{{}, {1.0f, 1.0f, 1.0f, 1.0f}};

  CascadesPushConstant push_constant{.sdf_id = SDFImage,
                                     .cascades_rw_id = CascadesImage,
                                     .cascades_id = CascadesTexture,
                                     .scene_rad_id = SceneRadianceTexture,
                                     .cascade_width = cascades_width_,
                                     .cascade_height = cascades_height_,
                                     .cascade_idx = 0,
                                     .cascade_count = config_.cascades.cascades,
                                     .base_probe_size = base_probe_size_,
                                     .base_spacing = base_probe_spacing_,
                                     .base_probe_dir_count = base_probe_size_ * base_probe_size_,
                                     .base_length = base_probe_length_,
                                     .light_id = LightImage,
                                     .scene_col_id = SceneColorTexture,
                                     .first = first};

  const uint32_t gx = (cascades_width_ + 15) / 16;
  const uint32_t gy = (cascades_height_ + 7) / 8;

  for (uint32_t i = config_.cascades.cascades; i-- > 0;) {
    std::string debug_name = std::format("Cascade {}", i);
    debug_label.pLabelName = debug_name.c_str();
    cmd.beginDebugUtilsLabelEXT(*debug_label);

    push_constant.cascade_idx = i;

    cmd.pushConstants(cascades_pipeline_layout_.get(), vk::ShaderStageFlagBits::eCompute, 0,
                      sizeof(CascadesPushConstant), &push_constant);


    cmd.dispatch(gx, gy, 1);
    cmd.pipelineBarrier2(vk::DependencyInfo{}.setImageMemoryBarriers(barrier));
    cmd.endDebugUtilsLabelEXT();
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

void Renderer2D::blur_pass(vk::CommandBuffer cmd)
{
  cmd.bindPipeline(vk::PipelineBindPoint::eCompute, blur_pipeline_.get());
  cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, blur_pipeline_layout_.get(), 0, 1, &bindless_descriptor_set_,
                         0, nullptr);

  auto barrier =
      vk::ImageMemoryBarrier2{}
          .setSrcStageMask(vk::PipelineStageFlagBits2::eComputeShader)
          .setSrcAccessMask(vk::AccessFlagBits2::eShaderStorageWrite)
          .setDstStageMask(vk::PipelineStageFlagBits2::eComputeShader)
          .setDstAccessMask(vk::AccessFlagBits2::eShaderSampledRead | vk::AccessFlagBits2::eShaderStorageWrite)
          .setOldLayout(vk::ImageLayout::eGeneral)
          .setNewLayout(vk::ImageLayout::eGeneral)
          .setSrcQueueFamilyIndex(vk::QueueFamilyIgnored)
          .setDstQueueFamilyIndex(vk::QueueFamilyIgnored)
          .setSubresourceRange({vk::ImageAspectFlagBits::eColor, 0, vk::RemainingMipLevels, 0, 1})
          .setImage(blur_image_->image());

  BlurPushConstant blur_push_constant{.light_id = LightTexture,
                                      .blur_id = BlurTexture,
                                      .blur_rw_id = BlurImage,
                                      .sampler_id = LinearSampler,
                                      .pass = 0,
                                      .sample_count = static_cast<uint32_t>(gaussian_weights_.size()),
                                      .weights_id = GaussianWeightsBuffer,
                                      .offsets_id = GaussianOffsetsBuffer};

  cmd.pushConstants(blur_pipeline_layout_.get(), vk::ShaderStageFlagBits::eCompute, 0, sizeof(BlurPushConstant),
                    &blur_push_constant);

  uint32_t gx = ((config_.scene_size.width >> 1) + 7) / 8;
  uint32_t gy = ((config_.scene_size.height >> 1) + 7) / 8;
  cmd.dispatch(gx, gy, 1);

  cmd.pipelineBarrier2(vk::DependencyInfo{}.setImageMemoryBarriers(barrier));

  blur_push_constant.pass = 1;
  cmd.pushConstants(blur_pipeline_layout_.get(), vk::ShaderStageFlagBits::eCompute, 0, sizeof(BlurPushConstant),
                    &blur_push_constant);

  cmd.dispatch(gx, gy, 1);

  barrier.setDstAccessMask(vk::AccessFlagBits2::eShaderSampledRead);
  barrier.subresourceRange.setBaseArrayLayer(1);
  cmd.pipelineBarrier2(vk::DependencyInfo{}.setImageMemoryBarriers(barrier));

  barrier.setImage(light_image_->image());
  barrier.subresourceRange.setBaseArrayLayer(0);
  barrier.setSrcAccessMask(vk::AccessFlagBits2::eShaderSampledRead);
  barrier.setDstAccessMask(vk::AccessFlagBits2::eShaderStorageWrite);
  cmd.pipelineBarrier2(vk::DependencyInfo{}.setImageMemoryBarriers(barrier));

  cmd.bindPipeline(vk::PipelineBindPoint::eCompute, blur_resolve_pipeline_.get());
  cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, blur_resolve_pipeline_layout_.get(), 0, 1,
                         &bindless_descriptor_set_, 0, nullptr);

  constexpr BlurResolvePushConstant resolve_push_constant{
      .light_rw_id = LightImage, .blur_id = BlurResolveTexture, .sampler_id = LinearSampler};

  cmd.pushConstants(blur_resolve_pipeline_layout_.get(), vk::ShaderStageFlagBits::eCompute, 0,
                    sizeof(BlurResolvePushConstant), &resolve_push_constant);

  gx = (config_.scene_size.width + 7) / 8;
  gy = (config_.scene_size.height + 7) / 8;
  cmd.dispatch(gx, gy, 1);

  barrier.setSrcAccessMask(vk::AccessFlagBits2::eShaderStorageWrite);
  barrier.setDstAccessMask(vk::AccessFlagBits2::eShaderSampledRead);
  cmd.pipelineBarrier2(vk::DependencyInfo{}.setImageMemoryBarriers(barrier));
}

void Renderer2D::composite_pass(vk::CommandBuffer cmd)
{
  cmd.bindPipeline(vk::PipelineBindPoint::eCompute, composite_pipeline_.get());
  cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, composite_pipeline_layout_.get(), 0, 1,
                         &bindless_descriptor_set_, 0, nullptr);

  const CompositePushConstant push_constant{.composite_id = CompositeImage,
                                            .light_id = LightImage,
                                            .sdf_id = SDFImage,
                                            .scene_col_id = SceneColorTexture,
                                            .scene_rad_id = SceneRadianceTexture,
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

  const BlitPushConstant push_constant{
      .composite_id = CompositeTexture,
      .aces = config_.composite.aces,
  };

  cmd.pushConstants(blit_pipeline_layout_.get(), vk::ShaderStageFlagBits::eFragment, 0, sizeof(BlitPushConstant),
                    &push_constant);

  cmd.draw(3, 1, 0, 0);
}

void Renderer2D::imgui_pass(vk::CommandBuffer cmd) const
{
  ImGui_ImplVulkan_RenderDrawData(info_.imgui_draw_data, cmd);
}

void Renderer2D::import_swapchain_images()
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

void Renderer2D::import_images()
{
  fwrk::ImageImportInfo image_import_info{
      .type = VK_IMAGE_TYPE_2D,
      .size = {.width = config_.scene_size.width, .height = config_.scene_size.height, .depth = 1},
      .format = VK_FORMAT_R8_UINT,
      .mip_count = 1,
      .layer_count = 1,
      .state = fwrk::PhysicalState::Undefined};

  if (draw_id_) {
    context_.update_image(draw_id_, image_import_info, draw_image_->image());
  } else {
    draw_id_ = context_.import_image(image_import_info, draw_image_->image());
  }

  image_import_info.format = VK_FORMAT_R8G8B8A8_SRGB;
  if (scene_color_id_) {
    context_.update_image(scene_color_id_, image_import_info, scene_color_image_->image());
  } else {
    scene_color_id_ = context_.import_image(image_import_info, scene_color_image_->image());
  }

  image_import_info.format = VK_FORMAT_B10G11R11_UFLOAT_PACK32;
  if (scene_radiance_id_) {
    context_.update_image(scene_radiance_id_, image_import_info, scene_radiance_image_->image());
  } else {
    scene_radiance_id_ = context_.import_image(image_import_info, scene_radiance_image_->image());
  }

  image_import_info.format = VK_FORMAT_R32G32B32A32_SINT;
  image_import_info.layer_count = 2;
  if (jfa_id_) {
    context_.update_image(jfa_id_, image_import_info, jfa_image_->image());
  } else {
    jfa_id_ = context_.import_image(image_import_info, jfa_image_->image());
  }
  image_import_info.layer_count = 1;

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

  if (blur_id_) {
    context_.update_image(blur_id_, image_import_info, blur_image_->image());
  } else {
    blur_id_ = context_.import_image(image_import_info, blur_image_->image());
  }

  if (composite_id_) {
    context_.update_image(composite_id_, image_import_info, composite_image_->image());
  } else {
    composite_id_ = context_.import_image(image_import_info, composite_image_->image());
  }

  image_import_info.size.width = cascades_width_;
  image_import_info.size.height = cascades_height_;
  image_import_info.layer_count = config_.cascades.cascades;
  if (cascades_id_) {
    context_.update_image(cascades_id_, image_import_info, cascades_image_->image());
  } else {
    cascades_id_ = context_.import_image(image_import_info, cascades_image_->image());
  }
}

void Renderer2D::create_images()
{
  device_.get().waitIdle();

  draw_image_.emplace(device_.get_allocator(), device_.get(),
                      ImageDesc{}
                          .set_extent(config_.scene_size.width, config_.scene_size.height)
                          .set_format(vk::Format::eR8Uint)
                          .set_usage(vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled)
                          .set_debug_name("Draw Image"));

  scene_color_image_.emplace(device_.get_allocator(), device_.get(),
                             ImageDesc{}
                                 .set_extent(config_.scene_size.width, config_.scene_size.height)
                                 .set_format(vk::Format::eR8G8B8A8Srgb)
                                 .set_usage(vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled)
                                 .set_debug_name("Scene Color Image"));

  scene_radiance_image_.emplace(
      device_.get_allocator(), device_.get(),
      ImageDesc{}
          .set_extent(config_.scene_size.width, config_.scene_size.height)
          .set_format(vk::Format::eB10G11R11UfloatPack32)
          .set_usage(vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled)
          .set_debug_name("Scene Radiance Image"));

  jfa_image_.emplace(device_.get_allocator(), device_.get(),
                     ImageDesc{}
                         .set_extent(config_.scene_size.width, config_.scene_size.height)
                         .set_format(vk::Format::eR32G32B32A32Sint)
                         .set_layers(2)
                         .set_usage(vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eColorAttachment)
                         .set_debug_name("JFA Images"));

  sdf_image_.emplace(device_.get_allocator(), device_.get(),
                     ImageDesc{}
                         .set_extent(config_.scene_size.width, config_.scene_size.height)
                         .set_format(vk::Format::eR32Sfloat)
                         .set_usage(vk::ImageUsageFlagBits::eStorage)
                         .set_debug_name("SDF Image"));

  light_image_.emplace(device_.get_allocator(), device_.get(),
                       ImageDesc{}
                           .set_extent(config_.scene_size.width, config_.scene_size.height)
                           .set_format(vk::Format::eR16G16B16A16Sfloat)
                           .set_usage(vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled)
                           .set_debug_name("Light Image"));

  blur_image_.emplace(device_.get_allocator(), device_.get(),
                      ImageDesc{}
                          .set_extent(config_.scene_size.width >> 1, config_.scene_size.height >> 1)
                          .set_format(vk::Format::eR16G16B16A16Sfloat)
                          .set_layers(2)
                          .set_usage(vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled)
                          .set_debug_name("Blur Image"));

  composite_image_.emplace(device_.get_allocator(), device_.get(),
                           ImageDesc{}
                               .set_extent(config_.scene_size.width, config_.scene_size.height)
                               .set_format(vk::Format::eR16G16B16A16Sfloat)
                               .set_usage(vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled)
                               .set_debug_name("Composite Image"));


  draw_view_ = draw_image_->create_image_view(device_.get(), vk::ImageAspectFlagBits::eColor);
  scene_color_view_ = scene_color_image_->create_image_view(device_.get(), vk::ImageAspectFlagBits::eColor);
  scene_radiance_view_ = scene_radiance_image_->create_image_view(device_.get(), vk::ImageAspectFlagBits::eColor);
  jfa_view_ =
      jfa_image_->create_image_view(device_.get(), vk::ImageAspectFlagBits::eColor, vk::ImageViewType::e2DArray);
  sdf_view_ = sdf_image_->create_image_view(device_.get(), vk::ImageAspectFlagBits::eColor);
  light_view_ = light_image_->create_image_view(device_.get(), vk::ImageAspectFlagBits::eColor);
  blur_array_view_ =
      blur_image_->create_image_view(device_.get(), vk::ImageAspectFlagBits::eColor, vk::ImageViewType::e2DArray);
  blur_view_ =
      blur_image_->create_image_view(device_.get(), vk::ImageAspectFlagBits::eColor, vk::ImageViewType::e2D, {}, 0);
  blur_resolve_view_ =
      blur_image_->create_image_view(device_.get(), vk::ImageAspectFlagBits::eColor, vk::ImageViewType::e2D, {}, 1);
  composite_view_ = composite_image_->create_image_view(device_.get(), vk::ImageAspectFlagBits::eColor);

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

  cascades_image_.emplace(device_.get_allocator(), device_.get(),
                          ImageDesc{}
                              .set_extent(cascades_width_, cascades_height_)
                              .set_format(vk::Format::eR16G16B16A16Sfloat)
                              .set_layers(config_.cascades.cascades)
                              .set_usage(vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled)
                              .set_debug_name("Cascade Images"));
  cascades_view_ =
      cascades_image_->create_image_view(device_.get(), vk::ImageAspectFlagBits::eColor, vk::ImageViewType::e2DArray);
}

void Renderer2D::update_descriptor()
{
  DescriptorWriter writer{};
  writer.add_image(0, DrawImage, vk::DescriptorType::eStorageImage, draw_view_.get(), vk::ImageLayout::eGeneral)
      .add_image(0, JFAImage, vk::DescriptorType::eStorageImage, jfa_view_.get(), vk::ImageLayout::eGeneral)
      .add_image(0, SDFImage, vk::DescriptorType::eStorageImage, sdf_view_.get(), vk::ImageLayout::eGeneral)
      .add_image(0, CascadesImage, vk::DescriptorType::eStorageImage, cascades_view_.get(), vk::ImageLayout::eGeneral)
      .add_image(0, LightImage, vk::DescriptorType::eStorageImage, light_view_.get(), vk::ImageLayout::eGeneral)
      .add_image(0, BlurImage, vk::DescriptorType::eStorageImage, blur_array_view_.get(), vk::ImageLayout::eGeneral)
      .add_image(0, CompositeImage, vk::DescriptorType::eStorageImage, composite_view_.get(), vk::ImageLayout::eGeneral)
      .add_image(2, DrawTexture, vk::DescriptorType::eSampledImage, draw_view_.get(),
                 vk::ImageLayout::eShaderReadOnlyOptimal)
      .add_image(2, SceneColorTexture, vk::DescriptorType::eSampledImage, scene_color_view_.get(),
                 vk::ImageLayout::eShaderReadOnlyOptimal)
      .add_image(2, SceneRadianceTexture, vk::DescriptorType::eSampledImage, scene_radiance_view_.get(),
                 vk::ImageLayout::eShaderReadOnlyOptimal)
      .add_image(2, CascadesTexture, vk::DescriptorType::eSampledImage, cascades_view_.get(),
                 vk::ImageLayout::eShaderReadOnlyOptimal)
      .add_image(2, LightTexture, vk::DescriptorType::eSampledImage, light_view_.get(),
                 vk::ImageLayout::eShaderReadOnlyOptimal)
      .add_image(2, BlurTexture, vk::DescriptorType::eSampledImage, blur_view_.get(),
                 vk::ImageLayout::eShaderReadOnlyOptimal)
      .add_image(2, BlurResolveTexture, vk::DescriptorType::eSampledImage, blur_resolve_view_.get(),
                 vk::ImageLayout::eShaderReadOnlyOptimal)
      .add_image(2, CompositeTexture, vk::DescriptorType::eSampledImage, composite_view_.get(),
                 vk::ImageLayout::eShaderReadOnlyOptimal);

  for (uint32_t i = 0; i < texture_images_.size(); i++) {
    writer.add_image(2, SpriteTextures + i, vk::DescriptorType::eSampledImage, texture_images_[i].second.get(),
                     vk::ImageLayout::eShaderReadOnlyOptimal);
  }

  writer.update(device_.get(), bindless_descriptor_set_);
}

void Renderer2D::upload_sprites() const
{
  const Frame& frame = frames_.at(current_frame_);

  std::memcpy(frame.sprites_buffer->get_mapped_data(), sprites_.data(), sprites_.size() * sizeof(SpriteGPU));
  frame.sprites_buffer->flush_allocation(0, sprites_.size() * sizeof(SpriteGPU));

  DescriptorWriter{}
      .add_buffer(1, SpriteBuffer, vk::DescriptorType::eStorageBuffer, frame.sprites_buffer->buffer())
      .update(device_.get(), bindless_descriptor_set_);
}

void Renderer2D::calculate_gaussian(const uint32_t radius, const float k)
{
  const float sigma = static_cast<float>(radius) / k;
  const float two_sigma_squared = 2.0f * sigma * sigma;

  float weight_sum = 0.0f;
  std::vector<float> weights;
  weights.reserve(static_cast<size_t>(radius) + 1);
  for (uint32_t i = 0; i <= radius; i++) {
    const float weight = std::exp(-static_cast<float>(i * i) / two_sigma_squared);
    weights.push_back(weight);
    weight_sum += weight * (i == 0 ? 1.0f : 2.0f);
  }
  for (float& weight: weights) {
    weight /= weight_sum;
  }

  gaussian_weights_.clear();
  gaussian_offsets_.clear();

  gaussian_weights_.push_back(weights[0]);
  gaussian_offsets_.emplace_back(0.0f);

  for (uint32_t i = 1; i <= radius; i += 2) {
    if (i + 1 <= radius) {
      const float w1 = weights[i];
      const float w2 = weights[i + 1];

      const float w3 = w1 + w2;
      const float offset = (static_cast<float>(i) * w1 + static_cast<float>(i + 1) * w2) / w3;

      gaussian_weights_.push_back(w3);
      gaussian_offsets_.push_back(offset);
    } else {
      gaussian_weights_.emplace_back(weights[i]);
      gaussian_offsets_.emplace_back(static_cast<float>(i));
    }
  }
}

std::pair<int32_t, int32_t> Renderer2D::screen_to_pos(const std::pair<int32_t, int32_t>& screen) const
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

  const float x_factor = (static_cast<float>(screen.first) - static_cast<float>(dst_off_x)) / static_cast<float>(dst_w);
  const float y_factor =
      (static_cast<float>(screen.second) - static_cast<float>(dst_off_y)) / static_cast<float>(dst_h);

  return {static_cast<int32_t>(x_factor * img_w), static_cast<int32_t>(y_factor * img_h)};
}
fwrk::Context Renderer2D::create_fwrk_context() const
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
