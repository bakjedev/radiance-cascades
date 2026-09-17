#include "application.hpp"

#include <cmath>
#include <iostream>
#include "resource/types/shader_resource.hpp"

struct QuitEvent {};

Application::Application(const uint32_t window_width, const uint32_t window_height) :
    window_(event_dispatcher_, window_width, window_height, "everything.. seems to be in order."),
    input_(event_dispatcher_), renderer_(window_, resource_manager_, file_system_)
{
}

Application::~Application() = default;

void Application::run()
{
  const uint64_t quit_event_listener =
      event_dispatcher_.listen<QuitEvent>([this](const QuitEvent&) { running = false; });

  file_system_.add_route("assets", "../assets");

  while (running) {
    input_.begin_frame();
    poll_events();
    input_.end_frame();

    if (input_.key_pressed(KeyboardKey::Space)) {
      std::cout << "Jump!\n";
    }
    if (input_.key_down(KeyboardKey::Escape)) {
      running = false;
    }

    if (input_.mouse_down(MouseButton::Left)) {
      auto [mouse_x, mouse_y] = std::pair{input_.mouse_x(), input_.mouse_y()};

      const auto w = static_cast<float>(window_.width());
      const auto h = static_cast<float>(window_.height());
      const auto sx = static_cast<float>(w) / 2 - static_cast<float>(h) / 2;

      const float u = (mouse_x - sx) / h;
      const float v = mouse_y / h;

      const auto px = static_cast<int32_t>(std::floor(u * 256)) - 1;
      const auto py = static_cast<int32_t>(std::floor(v * 256)) - 1;

      renderer_.plot({px, py});
    }

    renderer_.render();
  }

  event_dispatcher_.remove(quit_event_listener);
}

void Application::poll_events()
{
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
        event_dispatcher_.dispatch(
            MouseMotionEvent{event.motion.x, event.motion.y, event.motion.xrel, event.motion.yrel});
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
