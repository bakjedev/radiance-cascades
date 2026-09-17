#pragma once
#include <vulkan/vulkan.hpp>

class Device;

class Swapchain {
public:
  Swapchain(const Device& device, vk::Extent2D window_extent,
            vk::ImageUsageFlags image_usage = vk::ImageUsageFlagBits::eColorAttachment);
  ~Swapchain();

  void recreate(vk::Extent2D window_extent);

  [[nodiscard]] std::optional<uint32_t> acquire_next_image(vk::Semaphore signal);
  [[nodiscard]] bool present(vk::Queue present_queue, uint32_t image_index, vk::Semaphore wait);

  [[nodiscard]] vk::SwapchainKHR get() const { return swapchain_.get(); }
  [[nodiscard]] vk::Format format() const { return surface_format_.format; }
  [[nodiscard]] vk::Extent2D extent() const { return extent_; }
  [[nodiscard]] uint32_t image_count() const { return static_cast<uint32_t>(images_.size()); }
  [[nodiscard]] const auto& image_views() const { return image_views_; }
  [[nodiscard]] vk::ImageView image_view(const uint32_t idx) const { return image_views_[idx]; }
  [[nodiscard]] vk::Image image(const uint32_t idx) const { return images_[idx]; }

private:
  const Device* device_;

  vk::ImageUsageFlags image_usage_;
  vk::SurfaceFormatKHR surface_format_{};
  vk::PresentModeKHR present_mode_{};
  vk::Extent2D extent_{};
  vk::UniqueSwapchainKHR swapchain_;
  std::vector<vk::Image> images_;
  std::vector<vk::ImageView> image_views_;

  vk::UniqueSwapchainKHR old_swapchain_;

  void create(vk::Extent2D window_extent);
  void destroy_views();

  static vk::SurfaceFormatKHR choose_surface_format(const Device& device);
  static vk::PresentModeKHR choose_present_mode(const Device& device);
  static vk::Extent2D choose_extent(const vk::SurfaceCapabilitiesKHR& capabilities, vk::Extent2D window_extent);
};
