#pragma once
#include "vk_mem_alloc.h"
#include "framework/allocator.hpp"

struct FwrkAllocator : fwrk::Allocator {
    std::optional<fwrk::PhysicalImage> create_image( const fwrk::ImageCreateInfo& img_info ) override;
    std::optional<fwrk::PhysicalBuffer> create_buffer( const fwrk::BufferCreateInfo& buf_info ) override;
    void destroy_image( fwrk::PhysicalImage& img ) override;
    void destroy_buffer( fwrk::PhysicalBuffer& buf ) override;

    fwrk::flat_hash_map<VkImage, VmaAllocation> image_to_allocation;
    fwrk::flat_hash_map<VkBuffer, VmaAllocation> buffer_to_allocation;
    VmaAllocator allocator;

    explicit FwrkAllocator( VmaAllocator alc ) : allocator(alc) {
    }
};
