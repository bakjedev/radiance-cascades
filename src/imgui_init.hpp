#pragma once
#include "imgui_impl_sdl3.h"
#include "implot.h"

class ImGuiInit {
public:
  explicit ImGuiInit(SDL_Window* window)
  {
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImPlot::CreateContext();
    ImGui_ImplSDL3_InitForVulkan(window);
  }

  ~ImGuiInit()
  {
    ImGui_ImplSDL3_Shutdown();
    ImPlot::DestroyContext();
    ImGui::DestroyContext();
  }
};
