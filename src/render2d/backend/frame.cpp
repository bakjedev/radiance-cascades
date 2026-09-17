#include "frame.hpp"

#include "device.hpp"

void create_frame(Frame& frame, const Device& device)
{
  constexpr vk::SemaphoreCreateInfo semaphore_create_info{};
  vk::FenceCreateInfo fence_create_info{};
  fence_create_info.setFlags(vk::FenceCreateFlagBits::eSignaled);

  frame.image_available = device.get().createSemaphoreUnique(semaphore_create_info);
  frame.in_flight = device.get().createFenceUnique(fence_create_info);

  vk::CommandPoolCreateInfo command_pool_create_info{};
  command_pool_create_info.setQueueFamilyIndex(device.get_queue_family());
  command_pool_create_info.setFlags(vk::CommandPoolCreateFlagBits::eTransient);

  frame.command_pool = device.get().createCommandPoolUnique(command_pool_create_info);

  vk::CommandBufferAllocateInfo allocate_info{};
  allocate_info.setCommandPool(frame.command_pool.get());
  allocate_info.setLevel(vk::CommandBufferLevel::ePrimary);
  allocate_info.setCommandBufferCount(1);

  frame.command_buffer = std::move(device.get().allocateCommandBuffersUnique(allocate_info).front());
}
