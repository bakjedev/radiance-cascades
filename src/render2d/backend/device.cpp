#define VMA_IMPLEMENTATION
#include "device.hpp"

#include <algorithm>
#include <iostream>
#include <ranges>
#include <set>
#include <stdexcept>
#include <vector>

Device::Device(vk::Instance instance, vk::SurfaceKHR surface) : surface_(surface, instance)
{
  pick_physical_device(instance);
  find_queue_families();
  create_device();
  get_queues();
  create_allocator(instance);
}

Device::~Device()
{
  if (allocator_) {
    vmaDestroyAllocator(allocator_);
  }
}

bool Device::is_device_suitable(const vk::PhysicalDevice& device)
{
  if (device.getProperties().apiVersion < vk::ApiVersion13) {
    return false;
  }

  auto available_extensions = device.enumerateDeviceExtensionProperties();

  if (std::ranges::none_of(available_extensions, [](const vk::ExtensionProperties& properties) {
        return std::string_view(properties.extensionName.data()) == vk::KHRSwapchainExtensionName;
      })) {
    return false;
  }

  const auto features = device.getFeatures2<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan13Features>();

  if (!features.get<vk::PhysicalDeviceVulkan13Features>().dynamicRendering) {
    return false;
  }

  if (!features.get<vk::PhysicalDeviceVulkan13Features>().synchronization2) {
    return false;
  }

  return true;
}

void Device::pick_physical_device(vk::Instance instance)
{
  const auto devices = instance.enumeratePhysicalDevices();

  if (devices.empty()) {
    throw std::runtime_error("Failed to find GPUs with Vulkan support");
  }

  for (auto& device: devices) {
    if (is_device_suitable(device)) {
      physical_device_ = device;
      return;
    }
  }

  throw std::runtime_error("Failed to get suitable physical device");
}

void Device::find_queue_families()
{
  const auto queue_families = physical_device_.getQueueFamilyProperties();

  constexpr vk::QueueFlags required = vk::QueueFlagBits::eGraphics | vk::QueueFlagBits::eCompute;

  for (size_t idx = 0; idx < queue_families.size(); idx++) {
    if ((queue_families.at(idx).queueFlags & required) != required) continue;

    if (physical_device_.getSurfaceSupportKHR(static_cast<uint32_t>(idx), surface_.get()) == 0U) continue;

    queue_family_indices_.graphics = idx;
    queue_family_indices_.compute = idx;
    queue_family_indices_.transfer = idx;
    queue_family_indices_.present = idx;
    return;
  }
  throw std::runtime_error("Failed to get queue families");
}

void Device::create_device()
{
  constexpr float queue_priority = 1.0F;
  vk::DeviceQueueCreateInfo queue_create_info{};
  queue_create_info.setQueueFamilyIndex(*queue_family_indices_.graphics).setQueuePriorities(queue_priority);

  const std::vector device_extensions = {vk::KHRSwapchainExtensionName};

  vk::StructureChain<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan11Features,
                     vk::PhysicalDeviceVulkan13Features, vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>
      feature_chain;

  feature_chain.get<vk::PhysicalDeviceVulkan13Features>().synchronization2 = true;

  vk::DeviceCreateInfo create_info{};
  create_info.setPNext(&feature_chain.get<vk::PhysicalDeviceFeatures2>());
  create_info.setQueueCreateInfos(queue_create_info);
  create_info.setPEnabledExtensionNames(device_extensions);

  device_ = physical_device_.createDeviceUnique(create_info);
}

void Device::get_queues() { graphics_queue_ = device_->getQueue(*queue_family_indices_.graphics, 0); }

void Device::create_allocator(vk::Instance instance)
{
  VmaAllocatorCreateInfo info = {};
  info.physicalDevice = physical_device_;
  info.device = device_.get();
  info.instance = instance;

  if (vmaCreateAllocator(&info, &allocator_) != VK_SUCCESS) {
    throw std::runtime_error("Failed to create Vulkan Memory Allocator");
  }
}
