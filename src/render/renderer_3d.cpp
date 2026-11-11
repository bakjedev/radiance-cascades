#include "renderer_3d.hpp"

Renderer3D::Renderer3D(const RendererCreateInfo& create_info) : backend_(create_info) {}

void Renderer3D::render([[maybe_unused]] RenderInfo info) {}

void Renderer3D::imgui_new_frame() { RenderBackend::imgui_new_frame(); }

void Renderer3D::request_recompile() { backend_.request_recompile(); }

fwrk::Graph::FrameTimings Renderer3D::get_timings() { return backend_.get_timings(); }
