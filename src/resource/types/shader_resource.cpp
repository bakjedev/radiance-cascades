#include "shader_resource.hpp"
#include "src/file_system.hpp"

ShaderResource ShaderResourceLoader::operator()(const std::string& path) const
{
  const auto full_path_str = file_system->resolve(path);
  if (!full_path_str) throw std::runtime_error("Couldn't resolve image resource path");
  const auto full_path = std::filesystem::weakly_canonical(full_path_str.value);

  auto data = file_system->read_binary(full_path.c_str());
  if (!data) {
    throw std::runtime_error("Failed to load shader: " + full_path.string());
  }
  return ShaderResource{.code = std::move(*data)};
}
