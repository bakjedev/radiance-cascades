#include "gltf_resource.hpp"
#include <cmath>
#include <fastgltf/core.hpp>
#include <fastgltf/tools.hpp>
#include <fastgltf/types.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <stdexcept>
#include "src/file_system.hpp"
#include "src/resource/resource_manager.hpp"
#include "src/resource/types/image_resource.hpp"

GLTFResource GLTFResourceLoader::operator()(const std::string& path) const
{
  GLTFResource result{};

  const auto full_path_str = file_system->resolve(path);
  if (!full_path_str) throw std::runtime_error("Couldn't resolve image resource path");
  const auto full_path = std::filesystem::path(full_path_str.value);

  auto data = fastgltf::GltfDataBuffer::FromPath(full_path);
  if (data.error() != fastgltf::Error::None) {
    throw std::runtime_error("The glTF file couldn't be loaded, or the buffer could not be allocated.");
  }

  fastgltf::Parser parser;

  auto asset = parser.loadGltf(data.get(), full_path.parent_path(), fastgltf::Options::None);
  if (const auto error = asset.error(); error != fastgltf::Error::None) {
    throw std::runtime_error("Some error occurred while reading the buffer, parsing the JSON, or validating the data.");
  }
  //
  // for (const auto& image: asset->images) {
  // }
  //
  // for (const auto& sampler: asset->samplers) {
  // }

  for (const auto& texture: asset->textures) {
    if (!texture.imageIndex || !texture.samplerIndex) continue;
    result.textures.emplace_back(*texture.imageIndex, *texture.samplerIndex);
  }
  //
  // for (const auto& material: asset->materials) {
  // }
  //
  // for (const auto& mesh: asset->meshes) {
  // }

  for (const auto& light: asset->lights) {
    result.lights.emplace_back(static_cast<GLTFResource::Light::Type>(light.type), glm::make_vec3(light.color.data()),
                               light.intensity, light.range.value_or(0.0f), light.innerConeAngle.value_or(0.0f),
                               light.outerConeAngle.value_or(0.0f));
  }

  for (const auto& camera: asset->cameras) {
    std::visit(fastgltf::visitor{
                   [&](const fastgltf::Camera::Orthographic& orthographic) {
                     glm::mat4 matrix{1.0f};
                     matrix[0][0] = 1.0f / orthographic.xmag;
                     matrix[1][1] = 1.0f / orthographic.ymag;
                     matrix[2][2] = 2.0f / (orthographic.znear - orthographic.zfar);
                     matrix[3][2] = (orthographic.zfar + orthographic.znear) / (orthographic.znear - orthographic.zfar);
                     result.cameras.emplace_back(matrix);
                   },
                   [&](const fastgltf::Camera::Perspective& perspective) {
                     glm::mat4 matrix{0.0f};

                     const float aspect_ratio = perspective.aspectRatio.value_or(16.0f / 9.0f);
                     matrix[0][0] = 1.0f / (aspect_ratio * std::tan(0.5f * perspective.yfov));
                     matrix[1][1] = 1.0f / std::tan(0.5f * perspective.yfov);
                     matrix[2][3] = -1.0f;

                     if (perspective.zfar.has_value()) {
                       matrix[2][2] = (*perspective.zfar + perspective.znear) / (perspective.znear - *perspective.zfar);
                       matrix[3][2] =
                           (2.0f * *perspective.zfar * perspective.znear) / (perspective.znear - *perspective.zfar);
                     } else {
                       matrix[2][2] = -1.0f;
                       matrix[3][2] = -2.0f * perspective.znear;
                     }

                     result.cameras.emplace_back(matrix);
                   }},
               camera.camera);
  }

  fastgltf::iterateSceneNodes(asset.get(), asset->defaultScene.value_or(0), fastgltf::math::fmat4x4(),
                              [&](fastgltf::Node& node, const fastgltf::math::fmat4x4& world) {
                                result.nodes.emplace_back(std::move(node.name), glm::make_mat4(world.data()),
                                                          node.meshIndex.value_or(UINT32_MAX),
                                                          node.lightIndex.value_or(UINT32_MAX),
                                                          node.cameraIndex.value_or(UINT32_MAX));
                              });

  return result;
}
