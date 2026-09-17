#pragma once
#include <vulkan/vulkan.hpp>

class Instance {
public:
  Instance();

  [[nodiscard]] vk::Instance get() const { return instance_.get(); }

private:
  vk::UniqueInstance instance_;
  vk::UniqueDebugUtilsMessengerEXT messenger_;
};
