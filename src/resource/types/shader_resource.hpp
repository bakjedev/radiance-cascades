#pragma once
#include <string>
#include <vector>

class FileSystem;

struct ShaderResource {
  std::vector<std::byte> code;
};


struct ShaderResourceLoader {
  ShaderResource operator()(const std::string& path) const;

  FileSystem* file_system;
};
