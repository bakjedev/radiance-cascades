#include "user_interface.hpp"
#include "imgui_impl_sdl3.h"
#include "implot.h"
#include "render2d/renderer_2d.hpp"

UserInterface::UserInterface(SDL_Window* window)
{
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImPlot::CreateContext();
  ImGui_ImplSDL3_InitForVulkan(window);
}

UserInterface::~UserInterface()
{
  ImGui_ImplSDL3_Shutdown();
  ImPlot::DestroyContext();
  ImGui::DestroyContext();
}

ImDrawData* UserInterface::render(Renderer2D& renderer)
{
  ImGui_ImplSDL3_NewFrame();

  ImGui::NewFrame();

  config_panel(renderer);
  performance_panel(renderer);

  ImGui::Render();

  return ImGui::GetDrawData();
}

void UserInterface::process_event(const SDL_Event& sdl_event) { ImGui_ImplSDL3_ProcessEvent(&sdl_event); }

bool UserInterface::consuming() { return ImGui::GetIO().WantCaptureMouse; }

void UserInterface::config_panel(Renderer2D& renderer)
{
  auto& cfg = renderer.config();
  ImGui::Begin("Config");

  constexpr uint32_t min_draw_size = 1;
  constexpr uint32_t max_draw_size = 100;
  ImGui::SliderScalar("Draw size", ImGuiDataType_U32, &cfg.drawing.size, &min_draw_size, &max_draw_size);

  ImGui::Text("Scene size");
  ImGui::SameLine();
  ImGui::PushItemWidth(80.0);
  ImGui::InputScalar("W", ImGuiDataType_U32, &cfg.scene_size.width);
  ImGui::SameLine();
  ImGui::InputScalar("H", ImGuiDataType_U32, &cfg.scene_size.height);
  ImGui::PopItemWidth();

  constexpr uint32_t min_cascades = 1;
  constexpr uint32_t max_cascades = 10;
  ImGui::SliderScalar("Cascades", ImGuiDataType_U32, &cfg.cascades.cascades, &min_cascades, &max_cascades);

  ImGui::InputFloat("Base Spacing", &cfg.cascades.base_spacing);
  ImGui::InputFloat("Base Interval", &cfg.cascades.base_interval);
  ImGui::InputFloat("Base Length", &cfg.cascades.base_length);
  ImGui::SliderScalar("Bounces", ImGuiDataType_U32, &cfg.cascades.bounce, &min_cascades, &max_cascades);

  auto& materials = renderer.materials();
  ImGui::Text("Materials:");
  for (size_t i = 0; i < materials.size(); i++) {
    const Renderer2D::Material& material = materials.at(i);
    const ImVec4 color{material.color[0], material.color[1], material.color[2], 1.0f};
    const ImVec4 radiance{material.radiance[0], material.radiance[1], material.radiance[2], 1.0f};
    const bool has_radiance = !(radiance.x == 0.0f && radiance.y == 0.0f && radiance.z == 0.0f);
    if (has_radiance) {
      ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 3.0f);
    }
    if (i > 0) ImGui::SameLine();
    if (ImGui::ColorButton((std::string("mat") + std::to_string(i)).c_str(), has_radiance ? radiance : color,
                           ImGuiColorEditFlags_NoTooltip)) {
      cfg.drawing.material = static_cast<uint8_t>(i);
    }

    if (has_radiance) {
      ImGui::PopStyleVar();

      if (ImGui::IsItemHovered()) {
        if (ImGui::BeginTooltip()) {
          ImGui::Text("Base color:");
          ImGui::SameLine();
          ImGui::ColorButton((std::string("mattooltip") + std::to_string(i)).c_str(), color,
                             ImGuiColorEditFlags_NoTooltip);
          ImGui::EndTooltip();
        }
      }
    }
  }

  ImGui::SliderFloat("Rim", &cfg.composite.rim, 0.0f, 100.0f);
  constexpr uint32_t blur_step = 1;
  ImGui::InputScalar("Blur Radius", ImGuiDataType_U32, &cfg.composite.blur_radius, &blur_step);

  if (ImGui::Button("Recompile framework")) {
    renderer.request_recompile();
  }
  ImGui::End();
}

void UserInterface::performance_panel(Renderer2D& renderer)
{
  auto ns_to_ms = [](const uint64_t ns) { return static_cast<float>(ns) / 1000000.0f; };
  const fwrk::Graph::FrameTimings timings = renderer.get_timings();

  for (auto& [_, times]: perf_passes_) times[perf_head_] = 0.0f;
  for (const auto& pass_timing: timings.passes) {
    const float ms = ns_to_ms(pass_timing.end - pass_timing.begin);

    PerfPass* pass = nullptr;
    for (auto& perf_pass: perf_passes_) {
      if (perf_pass.first == pass_timing.name) {
        pass = &perf_pass;
        break;
      }
    }
    if (!pass) {
      pass = &perf_passes_.emplace_back(pass_timing.name, Timings{});
    }

    pass->second[perf_head_] = ms;
  }

  std::erase_if(perf_passes_, [](const PerfPass& pass) {
    return std::ranges::all_of(pass.second, [](const float time) { return time == 0.0f; });
  });

  const float frame_ms = ns_to_ms(timings.end - timings.begin);
  perf_frame_[perf_head_] = frame_ms;
  perf_head_ = (perf_head_ + 1) % PerfHistory;

  constexpr auto xs = [] {
    Timings arr{};
    for (uint32_t i = 0; i < PerfHistory; i++) arr[i] = static_cast<float>(i);
    return arr;
  }();

  ImGui::Begin("Performance");
  ImPlot::PushStyleVar(ImPlotStyleVar_FitPadding, ImVec2(0.0f, 0.1f));
  if (ImPlot::BeginPlot("Pass timings")) {
    ImPlot::SetupAxes("frame", "time (ms)",
                      ImPlotAxisFlags_NoTickLabels | ImPlotAxisFlags_LockMin | ImPlotAxisFlags_LockMax,
                      ImPlotAxisFlags_AutoFit);
    ImPlot::SetupAxisLimits(ImAxis_X1, 0, PerfHistory - 1, ImPlotCond_Always);
    ImPlot::SetupAxisFormat(ImAxis_Y1, "%.2f ms");
    ImPlot::SetupLegend(ImPlotLocation_South, ImPlotLegendFlags_Outside);

    Timings ys1{0.0f};
    Timings ys2{0.0f};

    for (const auto& [name, pass_times]: perf_passes_) {
      for (uint32_t i = 0; i < PerfHistory; i++) {
        ys2[i] = ys1[i] + pass_times[(perf_head_ + i) % PerfHistory];
      }

      uint32_t latest = (perf_head_ + PerfHistory - 1) % PerfHistory;
      const std::string label = std::format("{:10} {:6.2f} ms###{}", name, pass_times[latest], name);

      const ImPlotSpec spec{ImPlotProp_FillColor, IMPLOT_AUTO_COL, ImPlotProp_FillAlpha, 0.85f};
      ImPlot::PlotShaded(label.c_str(), xs.data(), ys1.data(), ys2.data(), PerfHistory, spec);

      ys1.swap(ys2);
    }

    constexpr float sixty_fps = 1000.0f / 60.0f;
    const ImPlotSpec inf_spec{ImPlotProp_Flags, static_cast<int>(ImPlotInfLinesFlags_Horizontal) |
                                                    static_cast<int>(ImPlotItemFlags_NoLegend)};
    ImPlot::PlotInfLines("16.677ms", &sixty_fps, 1, inf_spec);

    ImPlot::EndPlot();
  }
  ImPlot::PopStyleVar();
  ImGui::End();
}
