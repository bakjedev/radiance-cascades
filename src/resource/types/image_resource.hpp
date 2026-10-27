#pragma once
#include <cstdint>
#include <string>
#include <vector>

class FileSystem;

struct ImageResource {
  std::vector<uint8_t> pixels;
  std::pair<uint32_t, uint32_t> size;
};


struct ImageResourceLoader {
  ImageResource operator()(const std::string& path) const;

  FileSystem* file_system;
};
