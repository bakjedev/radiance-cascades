#include "frame.hpp"

#include "device.hpp"

void create_frame(Frame& frame, const Device& device, const vk::DeviceSize sprites_buffer_size)
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

  frame.sprites_buffer.emplace(
      device.get_allocator(), device.get(),
      BufferDesc{}
          .set_size(sprites_buffer_size)
          .set_usage(vk::BufferUsageFlagBits::eStorageBuffer)
          .set_allocate_flags(VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT)
          .set_debug_name("Sprites"));
}
