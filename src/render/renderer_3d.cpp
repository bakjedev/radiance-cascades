#include "renderer_3d.hpp"

#include "imgui_impl_vulkan.h"

Renderer3D::Renderer3D(const RendererCreateInfo& create_info) : RenderBackend(create_info) {}

Renderer3D::~Renderer3D() { device_.get().waitIdle(); }

void Renderer3D::render([[maybe_unused]] RenderInfo info)
{
  info_ = std::move(info);
  if (!begin_frame({info_.window_size.first, info_.window_size.second})) {
    return;
  }

  if (should_compile_) compile();

  execute_frame();
  end_frame();
}

void Renderer3D::compile()
{
  fwrk::Graph& graph = context_.graph();

  graph.add_graphics_pass("ImGui")
      .set_color_attachment({
          .resource = {.id = swapchain_proxy_},
          .load_op = fwrk::LoadOp::Clear,
          .store_op = fwrk::StoreOp::Store,
      })
      .set_execute([this](vk::CommandBuffer cmd) { imgui_pass(cmd); });

  graph.set_image_end_state(swapchain_proxy_,
                            {VK_ACCESS_2_NONE, VK_PIPELINE_STAGE_2_NONE, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR});

  graph.compile();
}

void Renderer3D::imgui_pass(vk::CommandBuffer cmd) const
{
  ImGui_ImplVulkan_RenderDrawData(info_.imgui_draw_data, cmd);
}
