#include "renderer_2d.hpp"

#include <SDL3/SDL_vulkan.h>

#include "backend/pipeline.hpp"
#include "src/window.hpp"
#include "src/resource/resource_manager.hpp"
#include "src/resource/types/shader_resource.hpp"

Renderer2D::Renderer2D( Window& window, EventDispatcher& event_dispatcher,
                        ResourceManager<ShaderResource>& resource_manager,
                        FileSystem& file_system ) : window_(window),
                                                    event_dispatcher_(event_dispatcher),
                                                    resource_manager_(resource_manager), file_system_(file_system),
                                                    device_(instance_.get(),
                                                            create_surface(window, instance_)),
                                                    swapchain_(device_, {
                                                                   window.width(), window.height()
                                                               }, vk::ImageUsageFlagBits::eColorAttachment |
                                                                  vk::ImageUsageFlagBits::eTransferDst),
                                                    fwrk_allocator_(device_.get_allocator()),
                                                    context_(device_.get(),
                                                             frames_in_flight, fwrk_allocator_) {
    for (Frame& frame : frames_) {
        create_frame(frame, device_);
    }

    submit_semaphores_.resize(swapchain_.image_count());
    constexpr vk::SemaphoreCreateInfo semaphore_create_info{};
    for (auto& semaphore : submit_semaphores_) {
        semaphore = device_.get().createSemaphoreUnique(semaphore_create_info);
    }

    constexpr fwrk::ImageCreateInfo scene_image_create_info{
        .type = VK_IMAGE_TYPE_2D,
        .size = VkExtent3D{.width = 256, .height = 256, .depth = 1},
        .format = VK_FORMAT_R16G16B16A16_SFLOAT,
        .flags = {},
        .mips = 1,
        .layers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_LINEAR,
        .usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT
    };

    scene_image_ = fwrk_allocator_.create_image(scene_image_create_info).value().handle;

    import_resources();
    swapchain_proxy_ = context_.create_proxy();

    auto shader_resource = resource_manager_.create_from_file<ShaderResource>(
        "basic.comp.spv", ShaderResourceLoader{&file_system_});

    shader_module_ = create_shader_module(device_.get(), shader_resource->code);

    pipeline_layout_ = create_pipeline_layout(device_.get(), {}, {});

    const ComputePipelineDesc desc{.module = shader_module_.get(), .layout = pipeline_layout_.get()};

    pipeline_ = create_compute_pipeline(device_.get(), desc);
}

Renderer2D::~Renderer2D() {
    device_.get().waitIdle();
    fwrk::PhysicalImage fake_physical_image{scene_image_, fwrk::PhysicalState::Undefined};
    fwrk_allocator_.destroy_image(fake_physical_image);
}

void Renderer2D::render() {
    if (begin_frame()) {
        run_frame();
        end_frame();
    }
}

bool Renderer2D::begin_frame() {
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

void Renderer2D::run_frame() {
    const Frame& frame = frames_.at(current_frame_);
    fwrk::Graph& graph = context_.graph();

    if (should_compile_) {
        graph.add_compute_pass().set_image_transfer_dst(
            {.resource = {.id = scene_image_import_}}).set_execute(
            [this]( vk::CommandBuffer cmd ) {
                constexpr vk::ClearColorValue color{0.0f, 1.0f, 0.0f, 10.0f};
                constexpr vk::ImageSubresourceRange range{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};

                cmd.clearColorImage(
                    scene_image_,
                    vk::ImageLayout::eTransferDstOptimal,
                    color,
                    range);
            });

        graph.add_compute_pass().set_image_transfer_src({.resource = {.id = scene_image_import_}}).
                set_image_transfer_dst({
                    .resource = {.id = swapchain_proxy_}
                }).set_execute([this]( vk::CommandBuffer cmd ) {
                    constexpr vk::ImageBlit blit{
                        {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
                        {{{0, 0, 0}, {256, 256, 1}}},
                        {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
                        {{{420, 0, 0}, {1500, 1080, 1}}}
                    };

                    cmd.blitImage(scene_image_, vk::ImageLayout::eTransferSrcOptimal,
                                  swapchain_.image(image_index_), vk::ImageLayout::eTransferDstOptimal, 1, &blit,
                                  vk::Filter::eNearest);
                });

        graph.set_image_end_state(swapchain_proxy_, {
                                      VK_ACCESS_2_NONE, VK_PIPELINE_STAGE_2_NONE,
                                      VK_IMAGE_LAYOUT_PRESENT_SRC_KHR
                                  });

        graph.compile();
        should_compile_ = false;
    }


    context_.update_proxy(swapchain_proxy_, swapchain_imports_[image_index_]);

    graph.execute(frame.command_buffer.get(), current_frame_);
}

void Renderer2D::end_frame() {
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

void Renderer2D::import_resources() {
    swapchain_imports_.resize(swapchain_.image_count());

    const fwrk::ImageImportInfo swapchain_image_info{
        .type = VK_IMAGE_TYPE_2D,
        .size = {swapchain_.extent().width, swapchain_.extent().height, 1},
        .format = static_cast<VkFormat>(swapchain_.format()),
        .state = fwrk::PhysicalState::Undefined
    };

    for (uint32_t i = 0; i < swapchain_.image_count(); i++) {
        fwrk::ResourceID& res = swapchain_imports_[i];
        if (res) {
            context_.update_image(res, swapchain_image_info, swapchain_.image(i));
        } else {
            res = context_.import_image(swapchain_image_info, swapchain_.image(i), "Swapchain image");
        }
    }

    constexpr fwrk::ImageImportInfo scene_image_info{
        .type = VK_IMAGE_TYPE_2D,
        .size = {.width = 256, .height = 256, .depth = 1},
        .format = VK_FORMAT_R16G16B16A16_SFLOAT,
        .state = fwrk::PhysicalState::Undefined
    };
    if (scene_image_import_) {
        context_.update_image(scene_image_import_, scene_image_info, scene_image_);
    } else {
        scene_image_import_ = context_.import_image(scene_image_info, scene_image_);
    }
}

VkSurfaceKHR Renderer2D::create_surface( const Window& window, const Instance& instance ) {
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    if (!SDL_Vulkan_CreateSurface(window.get(), instance.get(), nullptr, &surface)) {
        throw std::runtime_error("Failed to create Vulkan Surface with SDL");
    }
    return surface;
}
