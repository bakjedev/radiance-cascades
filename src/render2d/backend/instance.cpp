#include "instance.hpp"
#include <SDL3/SDL_vulkan.h>

VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

static VKAPI_ATTR vk::Bool32 VKAPI_CALL debug_callback(vk::DebugUtilsMessageSeverityFlagBitsEXT,
                                                       vk::DebugUtilsMessageTypeFlagsEXT,
                                                       const vk::DebugUtilsMessengerCallbackDataEXT* data, void*)
{
  std::fprintf(stderr, "vk: %s\n", data->pMessage);
  return VK_FALSE;
}

Instance::Instance()
{
  static vk::detail::DynamicLoader dl;
  VULKAN_HPP_DEFAULT_DISPATCHER.init(dl.getProcAddress<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr"));

  std::vector<const char*> extensions;
  std::vector<const char*> layers;

#ifndef NDEBUG
  layers.push_back("VK_LAYER_KHRONOS_validation");
#endif

  uint32_t sdl_instance_extension_count = 0;
  const auto* const sdl_instance_extensions = SDL_Vulkan_GetInstanceExtensions(&sdl_instance_extension_count);
  if (sdl_instance_extensions == nullptr) {
    throw std::runtime_error("Failed to get SDL Vulkan extensions");
  }

  extensions.reserve(sdl_instance_extension_count);
  for (uint32_t i = 0; i < sdl_instance_extension_count; i++) {
    extensions.push_back(sdl_instance_extensions[i]);
  }

  extensions.push_back(vk::EXTDebugUtilsExtensionName);

  vk::ApplicationInfo app_info{"app", 1, "engine", 1, vk::ApiVersion13};


  vk::DebugUtilsMessengerCreateInfoEXT debug_create_info{
      {},
      vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning | vk::DebugUtilsMessageSeverityFlagBitsEXT::eError,
      vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral | vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation |
          vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance,
      debug_callback};

  const vk::InstanceCreateInfo create_info{{}, &app_info, layers, extensions, &debug_create_info};

  instance_ = vk::createInstanceUnique(create_info);
  VULKAN_HPP_DEFAULT_DISPATCHER.init(*instance_);

  messenger_ = instance_->createDebugUtilsMessengerEXTUnique(debug_create_info);
}
