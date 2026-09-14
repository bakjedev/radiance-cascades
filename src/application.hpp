#pragma once
#include "sdl_init.hpp"
#include "window.hpp"
#include "event_dispatcher.hpp"
#include "input.hpp"
#include "render2d/renderer_2d.hpp"

class Application {
public:
    Application( uint32_t window_width, uint32_t window_height );

    ~Application();

    void run();

private:
    SDLInit sdl_init_;
    EventDispatcher event_dispatcher_;
    Window window_;
    Input input_;
    Renderer2D renderer_;
    bool running = true;

    void poll_events();
};
