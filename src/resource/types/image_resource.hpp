#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include "src/render/renderer_fwd.hpp"
#include "src/render/renderer_types.hpp"

class FileSystem;

struct ImageResource {
  GPUImage gpu_image;
  std::pair<uint32_t, uint32_t> size;
};


struct ImageResourceLoader {
  ImageResource operator()(const std::string& path) const;
  ImageResource operator()(std::vector<uint8_t>& pixels, const std::pair<uint32_t, uint32_t>& size) const;

  FileSystem* file_system;
  ActiveRenderer* renderer;
};
