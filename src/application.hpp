#pragma once
#include "event_dispatcher.hpp"
#include "file_system.hpp"
#include "input.hpp"
#include "render/active_renderer.hpp"
#include "render/renderer_fwd.hpp"
#include "resource/resource_manager.hpp"
#include "sdl_init.hpp"
#include "user_interface.hpp"
#include "window.hpp"

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
  FileSystem file_system_;
  ResourceManager resource_manager_;
  UserInterface user_interface_;
  ActiveRenderer renderer_;
  bool running = true;

  void poll_events();

  ActiveRenderer create_renderer();
};
