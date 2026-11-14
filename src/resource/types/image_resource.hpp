#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include "src/render/renderer_fwd.hpp"
#include "src/render/renderer_types.hpp"

class FileSystem;
class RenderBackend;

struct ImageResource {
  GPUImage gpu_image;
  std::pair<uint32_t, uint32_t> size;
};


struct ImageResourceLoader {
  ImageResource operator()(const std::string& path) const;
  ImageResource operator()(const std::vector<std::byte>& image_data) const;

  FileSystem* file_system;
  RenderBackend* renderer;
};
