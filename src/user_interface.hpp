#pragma once
#include <array>
#include <cstdint>
#include <string>
#include <vector>

struct ImDrawData;
struct SDL_Window;
class Renderer2D;
union SDL_Event;

class UserInterface {
public:
  explicit UserInterface(SDL_Window* window);
  ~UserInterface();

  ImDrawData* render(Renderer2D& renderer);

  static void process_event(const SDL_Event& sdl_event);
  [[nodiscard]] static bool consuming();

private:
  static void config_panel(Renderer2D& renderer);
  void performance_panel(Renderer2D& renderer);

  static constexpr uint32_t PerfHistory = 512;
  using Timings = std::array<float, PerfHistory>;
  using PerfPass = std::pair<std::string, Timings>;
  std::vector<PerfPass> perf_passes_;
  Timings perf_frame_{};
  uint32_t perf_head_{0};
};
