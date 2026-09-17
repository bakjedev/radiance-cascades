#pragma once
#include "event_dispatcher.hpp"
#include "file_system.hpp"
#include "input.hpp"
#include "render2d/renderer_2d.hpp"
#include "resource/resource_manager.hpp"
#include "sdl_init.hpp"
#include "window.hpp"

struct ShaderResource;

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
  ResourceManager<ShaderResource> resource_manager_;
  Renderer2D renderer_;
  bool running = true;

  void poll_events();
};
