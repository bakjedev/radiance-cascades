#pragma once
#include <vk_mem_alloc.h>
#include <vulkan/vulkan.hpp>

struct QueueFamilyIndices {
    std::optional<uint32_t> graphics;
    std::optional<uint32_t> compute;
    std::optional<uint32_t> transfer;
    std::optional<uint32_t> present;
};

class Device {
public:
    explicit Device( vk::Instance instance, vk::SurfaceKHR surface );
    ~Device();

    [[nodiscard]] vk::SurfaceKHR get_surface() const { return surface_.get(); }
    [[nodiscard]] const vk::PhysicalDevice& get_physical() const { return physical_device_; }
    [[nodiscard]] vk::Device get() const { return device_.get(); }
    [[nodiscard]] vk::Queue get_queue() const { return graphics_queue_; }

private:
    vk::UniqueSurfaceKHR surface_;
    vk::PhysicalDevice physical_device_;
    QueueFamilyIndices queue_family_indices_;
    vk::UniqueDevice device_;
    vk::Queue graphics_queue_;

    static bool is_device_suitable( const vk::PhysicalDevice& device );
    void pick_physical_device( vk::Instance instance );
    void find_queue_families();
    void create_device();
    void get_queues();
};
