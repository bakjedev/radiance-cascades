#pragma once
#include "backend/device.hpp"
#include "backend/frame.hpp"
#include "backend/instance.hpp"
#include "backend/swapchain.hpp"

class Window;
class EventDispatcher;

constexpr uint32_t frames_in_flight = 2;

class Renderer2D {
public:
    Renderer2D( Window& window, EventDispatcher& event_dispatcher );
    ~Renderer2D();

    bool begin_frame();
    void end_frame();

private:
    Window& window_;
    EventDispatcher& event_dispatcher_;

    Instance instance_;
    Device device_;
    Swapchain swapchain_;
    std::array<Frame, frames_in_flight> frames_;
    std::vector<vk::UniqueSemaphore> submit_semaphores_;

    uint32_t current_frame_{};
    uint32_t image_index_{};

    static VkSurfaceKHR create_surface( const Window& window, const Instance& instance );
};
