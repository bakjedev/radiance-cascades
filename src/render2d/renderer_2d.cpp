#include "renderer_2d.hpp"

#include <SDL3/SDL_vulkan.h>

#include "src/window.hpp"

std::optional<fwrk::PhysicalImage> FwrkAllocator::create_image( const fwrk::ImageCreateInfo& img_info ) {
    VmaAllocation allocation;
    VkImage image;

    vk::ImageCreateInfo create_info = {};
    create_info.setImageType(static_cast<vk::ImageType>(img_info.type));
    create_info.setFormat(static_cast<vk::Format>(img_info.format));
    create_info.setFlags(vk::ImageCreateFlags(img_info.flags));
    create_info.setExtent(img_info.size);
    create_info.setMipLevels(img_info.mips);
    create_info.setArrayLayers(img_info.layers);
    create_info.setSamples(static_cast<vk::SampleCountFlagBits>(img_info.samples));
    create_info.setTiling(static_cast<vk::ImageTiling>(img_info.tiling));
    create_info.setUsage(vk::ImageUsageFlags(img_info.usage));
    create_info.setSharingMode(vk::SharingMode::eExclusive);
    create_info.setInitialLayout(vk::ImageLayout::eUndefined);

    VmaAllocationCreateInfo alloc_info = {};
    alloc_info.usage = VMA_MEMORY_USAGE_AUTO;

    vmaCreateImage(allocator, reinterpret_cast<const VkImageCreateInfo*>(&create_info), &alloc_info, &image,
                   &allocation, nullptr);


    image_to_allocation[image] = allocation;
    return fwrk::PhysicalImage{image, fwrk::PhysicalState::Undefined};
}

std::optional<fwrk::PhysicalBuffer> FwrkAllocator::create_buffer( const fwrk::BufferCreateInfo& buf_info ) {
    VmaAllocation allocation;
    VkBuffer buffer;

    vk::BufferCreateInfo create_info = {};
    create_info.setSize(buf_info.size);
    create_info.setFlags(vk::BufferCreateFlags(buf_info.flags));
    create_info.setUsage(vk::BufferUsageFlags(buf_info.usage));

    VmaAllocationCreateInfo alloc_info = {};
    alloc_info.usage = VMA_MEMORY_USAGE_AUTO;

    vmaCreateBuffer(allocator, reinterpret_cast<const VkBufferCreateInfo*>(&create_info), &alloc_info, &buffer,
                    &allocation, nullptr);

    buffer_to_allocation[buffer] = allocation;
    return fwrk::PhysicalBuffer{buffer, fwrk::PhysicalState::Undefined};
}

void FwrkAllocator::destroy_image( fwrk::PhysicalImage& img ) {
    vmaDestroyImage(allocator, img.handle, image_to_allocation[img.handle]);
}

void FwrkAllocator::destroy_buffer( fwrk::PhysicalBuffer& buf ) {
    vmaDestroyBuffer(allocator, buf.handle, buffer_to_allocation[buf.handle]);
}

Renderer2D::Renderer2D( Window& window, EventDispatcher& event_dispatcher ) : window_(window),
                                                                              event_dispatcher_(event_dispatcher),
                                                                              device_(instance_.get(),
                                                                                  create_surface(window, instance_)),
                                                                              swapchain_(device_, {
                                                                                      window.width(), window.height()
                                                                                  }),
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
}

Renderer2D::~Renderer2D() {
    device_.get().waitIdle();
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
        return false;
    }
    image_index_ = image_index.value();

    device_.get().resetFences(frame.in_flight.get());

    device_.get().resetCommandPool(frame.command_pool.get());
    frame.command_buffer->begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    return true;
}

void Renderer2D::run_frame() {
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
