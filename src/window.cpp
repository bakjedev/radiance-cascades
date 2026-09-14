#include "window.hpp"

#include <SDL3/SDL_video.h>

void WindowDestructor::operator()( SDL_Window* window ) const {
    SDL_DestroyWindow(window);
}

Window::Window( const uint32_t width, const uint32_t height, const char* title ) : handle_(
    SDL_CreateWindow(title, static_cast<int>(width), static_cast<int>(height),
                     SDL_WINDOW_RESIZABLE & SDL_WINDOW_VULKAN)) {
    if (!handle_) {
        throw std::runtime_error(std::string("Failed to create SDL window: ") + SDL_GetError());
    }
}
