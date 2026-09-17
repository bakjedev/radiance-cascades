#include "fwrk_allocator.hpp"
#include <vulkan/vulkan.hpp>

std::optional<fwrk::PhysicalImage> FwrkAllocator::create_image(const fwrk::ImageCreateInfo& img_info)
{
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

  vmaCreateImage(allocator, reinterpret_cast<const VkImageCreateInfo*>(&create_info), &alloc_info, &image, &allocation,
                 nullptr);


  image_to_allocation[image] = allocation;
  return fwrk::PhysicalImage{image, fwrk::PhysicalState::Undefined};
}

std::optional<fwrk::PhysicalBuffer> FwrkAllocator::create_buffer(const fwrk::BufferCreateInfo& buf_info)
{
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

void FwrkAllocator::destroy_image(fwrk::PhysicalImage& img)
{
  vmaDestroyImage(allocator, img.handle, image_to_allocation[img.handle]);
}

void FwrkAllocator::destroy_buffer(fwrk::PhysicalBuffer& buf)
{
  vmaDestroyBuffer(allocator, buf.handle, buffer_to_allocation[buf.handle]);
}
