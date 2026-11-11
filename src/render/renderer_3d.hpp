#pragma once
#include "render_backend.hpp"
#include "renderer.hpp"

class Renderer3D {
public:
  explicit Renderer3D(const RendererCreateInfo& create_info);
  ~Renderer3D();
  void render(RenderInfo info);
  static void imgui_new_frame();
  void request_recompile();
  [[nodiscard]] fwrk::Graph::FrameTimings get_timings();

private:
  RenderBackend backend_;
};
