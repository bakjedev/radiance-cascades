#include "application.hpp"

#include <iostream>

struct QuitEvent {
};

Application::Application( const uint32_t window_width, const uint32_t window_height ) : window_(
        window_width, window_height, "everything.. seems to be in order."), input_(event_dispatcher_),
    renderer_(window_, event_dispatcher_) {
}

Application::~Application() = default;

void Application::run() {
    event_dispatcher_.listen<QuitEvent>([this]( const QuitEvent& ) {
        running = false;
    });

    while (running) {
        input_.begin_frame();
        poll_events();
        input_.end_frame();

        if (input_.key_down(KeyboardKey::A)) {
            std::cout << "Hello world !\n";
        }
    }
}

void Application::poll_events() {
    SDL_Event event;
    while (SDL_PollEvent(&event)) {
        switch (event.type) {
            case SDL_EVENT_QUIT :
                event_dispatcher_.dispatch(QuitEvent{});
                break;
            default :
                break;
        }
    }
}
