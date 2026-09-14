#pragma once
#include <cstdint>
#include <memory>

struct SDL_Window;

struct WindowDestructor {
    void operator()( SDL_Window* window ) const;
};

class Window {
public:
    Window( uint32_t width, uint32_t height, const char* title );

    [[nodiscard]] SDL_Window* get() const { return handle_.get(); }

private:
    std::unique_ptr<SDL_Window, WindowDestructor> handle_;
};
