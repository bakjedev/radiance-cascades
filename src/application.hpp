#pragma once
#include "sdl_init.hpp"
#include "window.hpp"
#include "event_dispatcher.hpp"
#include "input.hpp"

class Application {
public:
    Application(uint32_t window_width, uint32_t window_height);

    ~Application();

    void run();

private:
    SDLInit sdl_init_;
    EventDispatcher event_dispatcher_;
    Window window_;
    Input input_;
    bool running = true;

    void poll_events();
};
