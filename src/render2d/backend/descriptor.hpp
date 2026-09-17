#pragma once
#include <deque>
#include <vulkan/vulkan.hpp>

struct DescriptorSetLayoutDesc {
  std::vector<vk::DescriptorSetLayoutBinding> bindings;
  vk::DescriptorSetLayoutCreateFlags flags{};

  DescriptorSetLayoutDesc& add_binding(const uint32_t binding, const vk::DescriptorType type,
                                       const vk::ShaderStageFlags stages, const uint32_t count = 1)
  {
    bindings.emplace_back(binding, type, count, stages, nullptr);
    return *this;
  }
};

inline vk::UniqueDescriptorSetLayout create_descriptor_set_layout(vk::Device device,
                                                                  const DescriptorSetLayoutDesc& desc)
{
  return device.createDescriptorSetLayoutUnique(
      vk::DescriptorSetLayoutCreateInfo{}.setFlags(desc.flags).setBindings(desc.bindings));
}

inline vk::UniqueDescriptorPool create_descriptor_pool(vk::Device device, std::span<const vk::DescriptorPoolSize> sizes,
                                                       const uint32_t max_sets,
                                                       const vk::DescriptorPoolCreateFlags flags = {})
{
  return device.createDescriptorPoolUnique(
      vk::DescriptorPoolCreateInfo{}.setFlags(flags).setMaxSets(max_sets).setPoolSizes(sizes));
}

class DescriptorWriter {
public:
  DescriptorWriter& add_image(const uint32_t binding, const vk::DescriptorType type, vk::ImageView view,
                              vk::ImageLayout layout, vk::Sampler sampler = {})
  {
    auto& info = image_infos_.emplace_back(sampler, view, layout);
    writes_.push_back(vk::WriteDescriptorSet{}.setDstBinding(binding).setDescriptorType(type).setImageInfo(info));
    return *this;
  }

  DescriptorWriter& add_buffer(const uint32_t binding, const vk::DescriptorType type, vk::Buffer buf,
                               vk::DeviceSize offset = 0, vk::DeviceSize range = vk::WholeSize)
  {
    auto& info = buffer_infos_.emplace_back(buf, offset, range);
    writes_.push_back(vk::WriteDescriptorSet{}.setDstBinding(binding).setDescriptorType(type).setBufferInfo(info));
    return *this;
  }

  void update(vk::Device device, vk::DescriptorSet set)
  {
    for (auto& write: writes_) write.setDstSet(set);
    device.updateDescriptorSets(writes_, {});
  }

  void clear()
  {
    image_infos_.clear();
    buffer_infos_.clear();
    writes_.clear();
  }

private:
  // deque to prevent invalidation on resize
  std::deque<vk::DescriptorImageInfo> image_infos_;
  std::deque<vk::DescriptorBufferInfo> buffer_infos_;

  std::vector<vk::WriteDescriptorSet> writes_;
};
