#include "renderer_2d.hpp"

#include <SDL3/SDL_vulkan.h>

#include "src/window.hpp"

Renderer2D::Renderer2D( Window& window, EventDispatcher& event_dispatcher ) : window_(window),
                                                                              event_dispatcher_(event_dispatcher),
                                                                              device_(instance_.get(),
                                                                                  create_surface(window, instance_)) {
}

VkSurfaceKHR Renderer2D::create_surface( const Window& window, const Instance& instance ) {
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    if (!SDL_Vulkan_CreateSurface(window.get(), instance.get(), nullptr, &surface)) {
        throw std::runtime_error("Failed to create Vulkan Surface with SDL");
    }
    return surface;
}
