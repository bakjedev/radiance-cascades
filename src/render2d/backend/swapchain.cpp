#include "swapchain.hpp"

#include "device.hpp"

Swapchain::Swapchain( const Device& device, vk::Extent2D window_extent ) : device_(&device) {
    create(window_extent);
}

Swapchain::~Swapchain() {
    destroy_views();
}

void Swapchain::recreate( vk::Extent2D window_extent ) {
    device_->get().waitIdle();

    old_swapchain_ = std::move(swapchain_);
    destroy_views();

    create(window_extent);

    old_swapchain_.reset();
}

std::optional<uint32_t> Swapchain::acquire_next_image( vk::Semaphore signal ) {
    try {
        auto [_, index] = device_->get().acquireNextImageKHR(swapchain_.get(), UINT64_MAX, signal, nullptr);
        return index;
    } catch (const vk::OutOfDateKHRError&) {
        return std::nullopt;
    }
}

bool Swapchain::present( vk::Queue present_queue, const uint32_t image_index, vk::Semaphore wait ) {
    vk::PresentInfoKHR info{};
    info.setWaitSemaphores(wait);
    info.setSwapchains(swapchain_.get());
    info.setImageIndices(image_index);

    try {
        const auto result = present_queue.presentKHR(info);
        return result != vk::Result::eSuboptimalKHR;
    } catch (const vk::OutOfDateKHRError&) {
        return false;
    }
}

void Swapchain::create( const vk::Extent2D window_extent ) {
    surface_format_ = choose_surface_format(*device_);
    present_mode_ = choose_present_mode(*device_);

    const auto capabilities = device_->get_physical().getSurfaceCapabilitiesKHR(device_->get_surface());

    extent_ = choose_extent(capabilities, window_extent);

    uint32_t image_count = capabilities.minImageCount + 1;
    if (capabilities.maxImageCount > 0 && image_count > capabilities.maxImageCount) {
        image_count = capabilities.maxImageCount;
    }

    vk::SwapchainCreateInfoKHR create_info{};
    create_info.setSurface(device_->get_surface());
    create_info.setMinImageCount(image_count);
    create_info.setImageFormat(surface_format_.format);
    create_info.setImageColorSpace(surface_format_.colorSpace);
    create_info.setImageExtent(extent_);
    create_info.setImageArrayLayers(1);
    create_info.setImageUsage(vk::ImageUsageFlagBits::eColorAttachment);
    create_info.setPreTransform(capabilities.currentTransform);
    create_info.setCompositeAlpha(vk::CompositeAlphaFlagBitsKHR::eOpaque);
    create_info.setPresentMode(present_mode_);
    create_info.setClipped(vk::True);
    create_info.oldSwapchain = swapchain_.get();
    create_info.setImageSharingMode(vk::SharingMode::eExclusive);

    swapchain_ = device_->get().createSwapchainKHRUnique(create_info);
    images_ = device_->get().getSwapchainImagesKHR(swapchain_.get());

    image_views_.reserve(images_.size());
    for (const auto& img : images_) {
        vk::ImageViewCreateInfo view_create_info{};
        view_create_info.setImage(img);
        view_create_info.setViewType(vk::ImageViewType::e2D);
        view_create_info.setFormat(surface_format_.format);
        view_create_info.setSubresourceRange({vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1});

        image_views_.push_back(device_->get().createImageView(view_create_info));
    }
}

void Swapchain::destroy_views() {
    if (!device_) return;
    for (auto view : image_views_) device_->get().destroyImageView(view);
    image_views_.clear();
}

vk::SurfaceFormatKHR Swapchain::choose_surface_format( const Device& device ) {
    const auto surface_formats = device.get_physical().getSurfaceFormatsKHR(device.get_surface());

    if (surface_formats.empty()) {
        throw std::runtime_error("Failed to find Vulkan surface formats");
    }

    for (const auto& surfaceFormat : surface_formats) {
        if (surfaceFormat.format == vk::Format::eB8G8R8A8Srgb &&
            surfaceFormat.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {
            return surfaceFormat;
        }
    }
    return surface_formats.front();
}

vk::PresentModeKHR Swapchain::choose_present_mode( const Device& device ) {
    const auto present_modes = device.get_physical().getSurfacePresentModesKHR(device.get_surface());

    if (present_modes.empty()) {
        throw std::runtime_error("Failed to find present modes");
    }

    for (const auto& present_mode : present_modes) {
        if (present_mode == vk::PresentModeKHR::eMailbox) {
            return present_mode;
        }
    }
    return vk::PresentModeKHR::eFifo;
}

vk::Extent2D Swapchain::choose_extent( const vk::SurfaceCapabilitiesKHR& capabilities, vk::Extent2D window_extent ) {
    if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
        return capabilities.currentExtent;
    }
    return {
        std::clamp(window_extent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width),
        std::clamp(window_extent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height),
    };
}
