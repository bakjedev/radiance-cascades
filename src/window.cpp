#include "window.hpp"

#include <SDL3/SDL_video.h>

#include "event_dispatcher.hpp"

void WindowDestructor::operator()( SDL_Window* window ) const {
    SDL_DestroyWindow(window);
}

Window::Window( EventDispatcher& event_dispatcher, const uint32_t width, const uint32_t height,
                const char* title ) : event_dispatcher_(&event_dispatcher), handle_(
                                          SDL_CreateWindow(title, static_cast<int>(width), static_cast<int>(height),
                                                           SDL_WINDOW_RESIZABLE | SDL_WINDOW_VULKAN)), width_(width),
                                      height_(height),
                                      resize_listener_(
                                          event_dispatcher_->listen<WindowResizeEvent>(
                                              [this]( const WindowResizeEvent& event ) { window_resize(event); })) {
    if (!handle_) {
        throw std::runtime_error(std::string("Failed to create SDL window: ") + SDL_GetError());
    }
}

Window::~Window() {
    event_dispatcher_->remove(resize_listener_);
}

void Window::window_resize( const WindowResizeEvent& event ) {
    width_ = static_cast<uint32_t>(event.width);
    height_ = static_cast<uint32_t>(event.height);
}
