#include "image_resource.hpp"
#include <stdexcept>
#include "src/file_system.hpp"
#include "src/render/active_renderer.hpp"
#include "stb_image.h"

ImageResource ImageResourceLoader::operator()(const std::string& path) const
{
  const auto full_path = file_system->resolve(path);
  if (!full_path) throw std::runtime_error("Couldn't resolve image resource path");

  int width, height, channels;
  stbi_uc* data = stbi_load(full_path.value.c_str(), &width, &height, &channels, STBI_rgb_alpha);
  if (!data) throw std::runtime_error("Failed to load image resource");

  const std::pair size = {static_cast<uint32_t>(width), static_cast<uint32_t>(height)};
  ImageResource res{
      .gpu_image = renderer->create_texture(std::vector(data, data + static_cast<size_t>(width * height * 4)), size),
      .size = size};

  stbi_image_free(data);

  return res;
}

ImageResource ImageResourceLoader::operator()(const std::vector<std::byte>& image_data) const
{
  int width, height, channels;
  stbi_uc* data = stbi_load_from_memory(reinterpret_cast<const stbi_uc*>(image_data.data()),
                                        static_cast<int>(image_data.size()), &width, &height, &channels, 4);


  const std::pair size = {static_cast<uint32_t>(width), static_cast<uint32_t>(height)};
  ImageResource res{
      .gpu_image = renderer->create_texture(std::vector(data, data + static_cast<size_t>(width * height * 4)), size),
      .size = size};

  stbi_image_free(data);

  return res;
}
