#pragma once
#include <vk_mem_alloc.h>
#include <vulkan/vulkan.hpp>

struct ImageDesc {
  vk::Format format = vk::Format::eUndefined;
  vk::Extent3D extent{};
  vk::ImageType type = vk::ImageType::e2D;
  uint32_t mip_levels = 1;
  uint32_t array_layers = 1;
  vk::SampleCountFlagBits samples = vk::SampleCountFlagBits::e1;
  vk::ImageUsageFlags usage{};
  vk::ImageTiling tiling = vk::ImageTiling::eOptimal;

  VmaMemoryUsage memory_usage = VMA_MEMORY_USAGE_AUTO;
  VmaAllocationCreateFlags alloc_flags{};

  ImageDesc& set_format(const vk::Format fmt)
  {
    format = fmt;
    return *this;
  }

  ImageDesc& set_extent(const uint32_t width, const uint32_t height, const uint32_t depth = 1)
  {
    extent.width = width;
    extent.height = height;
    extent.depth = depth;
    return *this;
  }

  ImageDesc& set_type(const vk::ImageType image_type)
  {
    type = image_type;
    return *this;
  }

  ImageDesc& set_mips(const uint32_t mips)
  {
    mip_levels = mips;
    return *this;
  }

  ImageDesc& set_layers(const uint32_t layers)
  {
    array_layers = layers;
    return *this;
  }

  ImageDesc& set_samples(const vk::SampleCountFlagBits sample_count)
  {
    samples = sample_count;
    return *this;
  }

  ImageDesc& set_usage(const vk::ImageUsageFlags usage_flags)
  {
    usage = usage_flags;
    return *this;
  }

  ImageDesc& set_tiling(const vk::ImageTiling image_tiling)
  {
    tiling = image_tiling;
    return *this;
  }
};

inline vk::ImageAspectFlags get_aspect_for_format(const vk::Format format)
{
  switch (format) {
    case vk::Format::eD16Unorm:
    case vk::Format::eD32Sfloat:
    case vk::Format::eX8D24UnormPack32:
      return vk::ImageAspectFlagBits::eDepth;
    case vk::Format::eD16UnormS8Uint:
    case vk::Format::eD24UnormS8Uint:
    case vk::Format::eD32SfloatS8Uint:
      return vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil;
    case vk::Format::eS8Uint:
      return vk::ImageAspectFlagBits::eStencil;
    default:
      return vk::ImageAspectFlagBits::eColor;
  }
}

class Image {
public:
  Image() = default;

  Image(VmaAllocator allocator, const ImageDesc& desc) : allocator_(allocator) { create(desc); }

  ~Image() { destroy(); }

  Image(const Image&) = delete;
  Image& operator=(const Image&) = delete;

  Image(Image&& other) noexcept { swap(other); }

  Image& operator=(Image&& other) noexcept
  {
    if (this != &other) {
      destroy();
      swap(other);
    }
    return *this;
  }

  [[nodiscard]] vk::Image image() const { return image_; }
  [[nodiscard]] VmaAllocation allocation() const { return allocation_; }
  [[nodiscard]] vk::Format format() const { return format_; }
  [[nodiscard]] vk::Extent3D extent() const { return extent_; }
  [[nodiscard]] uint32_t mip_levels() const { return mip_levels_; }
  [[nodiscard]] uint32_t array_layers() const { return array_layers_; }
  [[nodiscard]] bool valid() const { return image_; }

  [[nodiscard]] vk::UniqueImageView create_image_view(vk::Device device,
                                                      vk::ImageAspectFlags aspect = vk::ImageAspectFlagBits::eNone,
                                                      const vk::ImageViewType view_type = vk::ImageViewType::e2D) const
  {
    if (aspect == vk::ImageAspectFlagBits::eNone) {
      aspect = get_aspect_for_format(format_);
    }

    const vk::ImageSubresourceRange range{aspect, 0, mip_levels_, 0, array_layers_};

    return device.createImageViewUnique(
        vk::ImageViewCreateInfo{}.setImage(image_).setViewType(view_type).setFormat(format_).setSubresourceRange(
            range));
  }

private:
  void create(const ImageDesc& desc)
  {
    vk::ImageCreateInfo create_info{};
    create_info.setImageType(desc.type)
        .setFormat(desc.format)
        .setExtent(desc.extent)
        .setMipLevels(desc.mip_levels)
        .setArrayLayers(desc.array_layers)
        .setSamples(desc.samples)
        .setTiling(desc.tiling)
        .setUsage(desc.usage)
        .setSharingMode(vk::SharingMode::eExclusive)
        .setInitialLayout(vk::ImageLayout::eUndefined);

    VmaAllocationCreateInfo allocation_create_info{};
    allocation_create_info.usage = desc.memory_usage;
    allocation_create_info.flags = desc.alloc_flags;

    const VkImageCreateInfo create_info_c = create_info;
    VkImage image_c;
    if (vmaCreateImage(allocator_, &create_info_c, &allocation_create_info, &image_c, &allocation_, nullptr) !=
        VK_SUCCESS) {
      throw std::runtime_error("Failed to create image with VMA");
    }
    image_ = image_c;
    format_ = desc.format;
    extent_ = desc.extent;
    mip_levels_ = desc.mip_levels;
    array_layers_ = desc.array_layers;
  }

  void destroy()
  {
    if (image_) vmaDestroyImage(allocator_, image_, allocation_);
    allocator_ = nullptr;
    image_ = nullptr;
    allocation_ = nullptr;
  }

  void swap(Image& other) noexcept
  {
    std::swap(allocator_, other.allocator_);
    std::swap(image_, other.image_);
    std::swap(allocation_, other.allocation_);
    std::swap(format_, other.format_);
    std::swap(extent_, other.extent_);
    std::swap(mip_levels_, other.mip_levels_);
    std::swap(array_layers_, other.array_layers_);
  }


  VmaAllocator allocator_ = nullptr;

  vk::Image image_ = nullptr;
  VmaAllocation allocation_ = nullptr;
  vk::Format format_ = vk::Format::eUndefined;
  vk::Extent3D extent_{};
  uint32_t mip_levels_ = 1;
  uint32_t array_layers_ = 1;
};
