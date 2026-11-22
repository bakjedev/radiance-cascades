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

enum class ImageColorSpace {
  Linear,
  sRGB,
};

struct ImageResourceLoader {
  ImageResource operator()(const std::string& path, ImageColorSpace color_space = ImageColorSpace::Linear) const;
  ImageResource operator()(std::span<const std::byte> image_data,
                           ImageColorSpace color_space = ImageColorSpace::Linear) const;

  FileSystem* file_system;
  RenderBackend* renderer;
};
