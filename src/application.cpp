#include "application.hpp"

#include <iostream>

struct QuitEvent {
};

Application::Application( const uint32_t window_width, const uint32_t window_height ) : window_(event_dispatcher_,
        window_width, window_height, "everything.. seems to be in order."), input_(event_dispatcher_),
    renderer_(window_, event_dispatcher_) {
}

Application::~Application() = default;

void Application::run() {
    const uint64_t quit_event_listener = event_dispatcher_.listen<QuitEvent>([this]( const QuitEvent& ) {
        running = false;
    });

    while (running) {
        input_.begin_frame();
        poll_events();
        input_.end_frame();

        if (renderer_.begin_frame()) {
            renderer_.end_frame();
        }


        if (input_.key_pressed(KeyboardKey::Space)) {
            std::cout << "Jump!\n";
        }
    }

    event_dispatcher_.remove(quit_event_listener);
}

void Application::poll_events() {
    SDL_Event event;
    while (SDL_PollEvent(&event)) {
        switch (event.type) {
            case SDL_EVENT_QUIT:
                event_dispatcher_.dispatch(QuitEvent{});
                break;
            case SDL_EVENT_KEY_DOWN:
                event_dispatcher_.dispatch(KeyDownEvent{static_cast<KeyboardKey>(event.key.scancode)});
                break;
            case SDL_EVENT_KEY_UP:
                event_dispatcher_.dispatch(KeyUpEvent{static_cast<KeyboardKey>(event.key.scancode)});
                break;
            case SDL_EVENT_MOUSE_BUTTON_DOWN:
                event_dispatcher_.dispatch(MouseDownEvent{static_cast<MouseButton>(event.button.button)});
                break;
            case SDL_EVENT_MOUSE_BUTTON_UP:
                event_dispatcher_.dispatch(MouseUpEvent{static_cast<MouseButton>(event.button.button)});
                break;
            case SDL_EVENT_MOUSE_MOTION:
                event_dispatcher_.dispatch(MouseMotionEvent{
                    event.motion.x, event.motion.y, event.motion.xrel, event.motion.yrel
                });
                break;
            case SDL_EVENT_MOUSE_WHEEL:
                event_dispatcher_.dispatch(MouseWheelEvent{static_cast<float>(event.wheel.integer_y)});
                break;
            case SDL_EVENT_WINDOW_RESIZED:
                event_dispatcher_.dispatch(WindowResizeEvent{event.window.data1, event.window.data2});
                break;
            default:
                break;
        }
    }
}
