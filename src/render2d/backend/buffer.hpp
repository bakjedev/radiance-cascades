#pragma once
#include <vk_mem_alloc.h>
#include <vulkan/vulkan.hpp>

struct BufferDesc {
  vk::BufferCreateFlags flags{};
  vk::DeviceSize size{};
  vk::BufferUsageFlags usage{};

  VmaMemoryUsage memory_usage = VMA_MEMORY_USAGE_AUTO;
  VmaAllocationCreateFlags alloc_flags{};

  BufferDesc& set_flags(const vk::BufferCreateFlags buffer_flags)
  {
    flags = buffer_flags;
    return *this;
  }

  BufferDesc& set_size(const vk::DeviceSize buffer_size)
  {
    size = buffer_size;
    return *this;
  }

  BufferDesc& set_usage(const vk::BufferUsageFlags buffer_usage)
  {
    usage = buffer_usage;
    return *this;
  }
};

class Buffer {
public:
  Buffer() = default;

  Buffer(VmaAllocator allocator, const BufferDesc& desc) : allocator_(allocator) { create(desc); }

  ~Buffer() { destroy(); }

  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;

  Buffer(Buffer&& other) noexcept { swap(other); }

  Buffer& operator=(Buffer&& other) noexcept
  {
    if (this != &other) {
      destroy();
      swap(other);
    }
    return *this;
  }

  [[nodiscard]] vk::Buffer buffer() const { return buffer_; }
  [[nodiscard]] VmaAllocation allocation() const { return allocation_; }
  [[nodiscard]] vk::DeviceSize size() const { return size_; }
  [[nodiscard]] bool valid() const { return buffer_; }

private:
  void create(const BufferDesc& desc)
  {
    vk::BufferCreateInfo create_info{};
    create_info.setFlags(desc.flags).setSize(desc.size).setUsage(desc.usage);

    VmaAllocationCreateInfo allocation_create_info{};
    allocation_create_info.usage = desc.memory_usage;
    allocation_create_info.flags = desc.alloc_flags;

    const VkBufferCreateInfo create_info_c = create_info;
    VkBuffer buffer_c;
    if (vmaCreateBuffer(allocator_, &create_info_c, &allocation_create_info, &buffer_c, &allocation_, nullptr) !=
        VK_SUCCESS) {
      throw std::runtime_error("Failed to create buffer with VMA");
    }
    buffer_ = buffer_c;
    size_ = desc.size;
  }

  void destroy()
  {
    if (buffer_) vmaDestroyBuffer(allocator_, buffer_, allocation_);
    allocator_ = nullptr;
    buffer_ = nullptr;
    allocation_ = nullptr;
  }

  void swap(Buffer& other) noexcept
  {
    std::swap(allocator_, other.allocator_);
    std::swap(buffer_, other.buffer_);
    std::swap(allocation_, other.allocation_);
    std::swap(size_, other.size_);
  }

  VmaAllocator allocator_ = nullptr;

  vk::Buffer buffer_ = nullptr;
  VmaAllocation allocation_ = nullptr;
  vk::DeviceSize size_{};
};
