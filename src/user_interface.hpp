#pragma once
#include <array>
#include <string>
#include <vector>

#include "render/renderer.hpp"

class Renderer2D;
class Renderer3D;
struct ImDrawData;
struct SDL_Window;
union SDL_Event;

class UserInterface {
public:
  explicit UserInterface(SDL_Window* window);
  ~UserInterface();

  template<Renderer T>
  ImDrawData* render(T& renderer)
  {
    T::imgui_new_frame();
    begin_frame();
    performance_panel(renderer.get_timings());
    if constexpr (std::is_same_v<T, Renderer2D>) {
      renderer_2d_panel(renderer);
    } else if constexpr (std::is_same_v<T, Renderer3D>) {
      renderer_3d_panel(renderer);
    }
    return end_frame();
  }

  static void process_event(const SDL_Event& sdl_event);
  [[nodiscard]] static bool consuming();

private:
  static void begin_frame();
  static ImDrawData* end_frame();
  static void renderer_2d_panel(Renderer2D& renderer);
  static void renderer_3d_panel(Renderer3D& renderer);
  void performance_panel(const fwrk::Graph::FrameTimings& timings);

  static constexpr uint32_t PerfHistory = 512;
  using Timings = std::array<float, PerfHistory>;
  using PerfPass = std::pair<std::string, Timings>;
  std::vector<PerfPass> perf_passes_;
  Timings perf_frame_{};
  uint32_t perf_head_{0};
};
