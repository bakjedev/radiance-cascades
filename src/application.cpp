#include "application.hpp"

struct QuitEvent {
};

Application::Application(const uint32_t window_width, const uint32_t window_height) : window_(
    window_width, window_height, "everything.. seems to be in order.") {
}

Application::~Application() = default;

void Application::run() {
    event_dispatcher_.listen<QuitEvent>([this](const QuitEvent &) {
        running = false;
    });

    while (running) {
        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            switch (event.type) {
                case SDL_EVENT_QUIT:
                    event_dispatcher_.dispatch(QuitEvent{});
                    break;
                default:
                    break;
            }
        }
    }
}
