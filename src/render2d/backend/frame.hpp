#pragma once
#include <vulkan/vulkan.hpp>

class Device;

struct Frame {
    vk::UniqueCommandPool command_pool;
    vk::UniqueCommandBuffer command_buffer;
    vk::UniqueSemaphore image_available;
    vk::UniqueFence in_flight;
};

void create_frame( Frame& frame, const Device& device );
