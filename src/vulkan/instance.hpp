#pragma once
#include <vulkan/vulkan.hpp>

class Instance {
public:
  explicit Instance(std::span<const char* const> instance_extensions);

  [[nodiscard]] vk::Instance get() const { return instance_.get(); }

private:
  vk::UniqueInstance instance_;
  vk::UniqueDebugUtilsMessengerEXT messenger_;
};
