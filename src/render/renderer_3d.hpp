#pragma once
#include "render_backend.hpp"
#include "renderer_interface.hpp"

class Renderer3D : public RenderBackend {
public:
  explicit Renderer3D(const RendererCreateInfo& create_info);
  ~Renderer3D();
  void render(RenderInfo info);

private:
  RenderInfo info_{};

  void compile();

  void imgui_pass(vk::CommandBuffer cmd) const;
};

static_assert(RendererInterface<Renderer3D>);
