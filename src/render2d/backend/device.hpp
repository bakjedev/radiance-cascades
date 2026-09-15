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
    [[nodiscard]] uint32_t get_queue_family() const { return queue_family_indices_.graphics.value_or(0); }
    [[nodiscard]] vk::Queue get_queue() const { return graphics_queue_; }
    [[nodiscard]] VmaAllocator get_allocator() const { return allocator_; }

private:
    vk::UniqueSurfaceKHR surface_;
    vk::PhysicalDevice physical_device_;
    QueueFamilyIndices queue_family_indices_;
    vk::UniqueDevice device_;
    vk::Queue graphics_queue_;
    VmaAllocator allocator_ = VK_NULL_HANDLE;

    static bool is_device_suitable( const vk::PhysicalDevice& device );
    void pick_physical_device( vk::Instance instance );
    void find_queue_families();
    void create_device();
    void get_queues();
    void create_allocator( vk::Instance instance );
};
