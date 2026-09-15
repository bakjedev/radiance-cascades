#pragma once
#include <cstdint>
#include <memory>

class EventDispatcher;
struct SDL_Window;

struct WindowResizeEvent {
    int32_t width;
    int32_t height;
};

struct WindowDestructor {
    void operator()( SDL_Window* window ) const;
};

class Window {
public:
    Window( EventDispatcher& event_dispatcher, uint32_t width, uint32_t height, const char* title );
    ~Window();

    [[nodiscard]] SDL_Window* get() const { return handle_.get(); }
    [[nodiscard]] uint32_t width() const { return width_; }
    [[nodiscard]] uint32_t height() const { return height_; }

private:
    EventDispatcher* event_dispatcher_;
    std::unique_ptr<SDL_Window, WindowDestructor> handle_;
    uint32_t width_;
    uint32_t height_;
    uint64_t resize_listener_;

    void window_resize( const WindowResizeEvent& event );
};
