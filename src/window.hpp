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
    [[nodiscard]] uint32_t width() const { return width_; }
    [[nodiscard]] uint32_t height() const { return height_; }

private:
    std::unique_ptr<SDL_Window, WindowDestructor> handle_;
    uint32_t width_;
    uint32_t height_;
};
