#include "renderer_2d.hpp"

#include <SDL3/SDL_vulkan.h>

#include "src/window.hpp"

Renderer2D::Renderer2D( Window& window, EventDispatcher& event_dispatcher ) : window_(window),
                                                                              event_dispatcher_(event_dispatcher),
                                                                              device_(instance_.get(),
                                                                                  create_surface(window, instance_)),
                                                                              swapchain_(device_, {
                                                                                      window.width(), window.height()
                                                                                  }) {
    for (Frame& frame : frames_) {
        create_frame(frame, device_);
    }

    submit_semaphores_.resize(swapchain_.image_count());
    constexpr vk::SemaphoreCreateInfo semaphore_create_info{};
    for (auto& semaphore : submit_semaphores_) {
        semaphore = device_.get().createSemaphoreUnique(semaphore_create_info);
    }
}

Renderer2D::~Renderer2D() {
    device_.get().waitIdle();
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
        return false;
    }
    image_index_ = image_index.value();

    device_.get().resetFences(frame.in_flight.get());

    device_.get().resetCommandPool(frame.command_pool.get());
    frame.command_buffer->begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    return true;
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
    }

    current_frame_ = (current_frame_ + 1) % frames_in_flight;
}

VkSurfaceKHR Renderer2D::create_surface( const Window& window, const Instance& instance ) {
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    if (!SDL_Vulkan_CreateSurface(window.get(), instance.get(), nullptr, &surface)) {
        throw std::runtime_error("Failed to create Vulkan Surface with SDL");
    }
    return surface;
}
