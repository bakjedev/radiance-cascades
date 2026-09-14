#pragma once
#include "backend/device.hpp"
#include "backend/instance.hpp"

class Window;
class EventDispatcher;

class Renderer2D {
public:
    Renderer2D( Window& window, EventDispatcher& event_dispatcher );

private:
    Window& window_;
    EventDispatcher& event_dispatcher_;

    Instance instance_;
    Device device_;

    static VkSurfaceKHR create_surface( const Window& window, const Instance& instance );
};
