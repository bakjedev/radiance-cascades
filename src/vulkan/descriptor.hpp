#pragma once
#include <deque>
#include <vulkan/vulkan.hpp>

struct DescriptorSetLayoutDesc {
  std::vector<vk::DescriptorSetLayoutBinding> bindings;
  std::vector<vk::DescriptorBindingFlags> binding_flags;
  vk::DescriptorSetLayoutCreateFlags layout_flags{};

  DescriptorSetLayoutDesc& add_binding(const uint32_t binding, const vk::DescriptorType type,
                                       const vk::ShaderStageFlags stages, const vk::DescriptorBindingFlags flags = {},
                                       const uint32_t count = 1)
  {
    bindings.emplace_back(binding, type, count, stages, nullptr);
    binding_flags.emplace_back(flags);
    return *this;
  }

  DescriptorSetLayoutDesc& set_layout_flags(const vk::DescriptorSetLayoutCreateFlags flags)
  {
    layout_flags = flags;
    return *this;
  }
};

inline vk::UniqueDescriptorSetLayout create_descriptor_set_layout(vk::Device device,
                                                                  const DescriptorSetLayoutDesc& desc)
{
  vk::DescriptorSetLayoutBindingFlagsCreateInfo binding_flags_create_info{};
  binding_flags_create_info.setBindingFlags(desc.binding_flags);

  return device.createDescriptorSetLayoutUnique(vk::DescriptorSetLayoutCreateInfo{}
                                                    .setFlags(desc.layout_flags)
                                                    .setBindings(desc.bindings)
                                                    .setPNext(&binding_flags_create_info));
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
  DescriptorWriter& add_image(const uint32_t binding, const uint32_t array_element, const vk::DescriptorType type,
                              vk::ImageView view, vk::ImageLayout layout, vk::Sampler sampler = {})
  {
    auto& info = image_infos_.emplace_back(sampler, view, layout);
    writes_.push_back(vk::WriteDescriptorSet{}
                          .setDstBinding(binding)
                          .setDstArrayElement(array_element)
                          .setDescriptorCount(1)
                          .setDescriptorType(type)
                          .setImageInfo(info));
    return *this;
  }

  DescriptorWriter& add_buffer(const uint32_t binding, const uint32_t array_element, const vk::DescriptorType type,
                               vk::Buffer buf, vk::DeviceSize offset = 0, vk::DeviceSize range = vk::WholeSize)
  {
    auto& info = buffer_infos_.emplace_back(buf, offset, range);
    writes_.push_back(vk::WriteDescriptorSet{}
                          .setDstBinding(binding)
                          .setDstArrayElement(array_element)
                          .setDescriptorCount(1)
                          .setDescriptorType(type)
                          .setBufferInfo(info));
    return *this;
  }

  DescriptorWriter& update(vk::Device device, vk::DescriptorSet set)
  {
    for (auto& write: writes_) write.setDstSet(set);
    device.updateDescriptorSets(writes_, {});
    return *this;
  }

  DescriptorWriter& clear()
  {
    image_infos_.clear();
    buffer_infos_.clear();
    writes_.clear();
    return *this;
  }

private:
  // deque to prevent invalidation on resize
  std::deque<vk::DescriptorImageInfo> image_infos_;
  std::deque<vk::DescriptorBufferInfo> buffer_infos_;

  std::vector<vk::WriteDescriptorSet> writes_;
};
