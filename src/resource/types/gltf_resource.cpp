#include "gltf_resource.hpp"
#include <cmath>
#include <fastgltf/core.hpp>
#include <fastgltf/tools.hpp>
#include <fastgltf/types.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <stdexcept>
#include "src/file_system.hpp"
#include "src/render/renderer_3d.hpp"
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

  auto asset = parser.loadGltf(data.get(), full_path.parent_path(), fastgltf::Options::LoadExternalBuffers);
  if (const auto error = asset.error(); error != fastgltf::Error::None) {
    throw std::runtime_error("Some error occurred while reading the buffer, parsing the JSON, or validating the data.");
  }

  for (const auto& image: asset->images) {
    std::visit(fastgltf::visitor{
                   [&]([[maybe_unused]] auto& arg) { result.images.emplace_back(); },
                   [&](fastgltf::sources::URI& uri) {
                     const auto resource = resource_manager->create_from_file<ImageResource>(
                         std::string(uri.uri->path()), ImageResourceLoader{file_system, renderer});
                     result.images.push_back(resource);
                   },
                   [&](fastgltf::sources::Vector& vector) {
                     const auto resource = resource_manager->load<ImageResource>(
                         std::format("image_{}_{}", full_path.string(), image.name),
                         ImageResourceLoader{file_system, renderer}, vector.bytes);
                     result.images.push_back(resource);
                   },
                   [&](const fastgltf::sources::BufferView& buffer_view) {
                     const auto& view = asset->bufferViews[buffer_view.bufferViewIndex];
                     auto& buffer = asset->buffers[view.bufferIndex];
                     std::visit(fastgltf::visitor{[&]([[maybe_unused]] auto& arg) { result.images.emplace_back(); },
                                                  [&](fastgltf::sources::Vector& vector) {
                                                    const auto resource = resource_manager->load<ImageResource>(
                                                        std::format("image_{}_{}", full_path.string(), image.name),
                                                        ImageResourceLoader{file_system, renderer}, vector.bytes);
                                                    result.images.push_back(resource);
                                                  }},
                                buffer.data);
                   },
               },
               image.data);
  }
  constexpr auto to_filter = [](const fastgltf::Filter filter) {
    switch (filter) {
      case fastgltf::Filter::Nearest:
        return Filter::Nearest;
      case fastgltf::Filter::Linear:
        return Filter::Linear;
      case fastgltf::Filter::NearestMipMapNearest:
        return Filter::NearestMipMapNearest;
      case fastgltf::Filter::LinearMipMapNearest:
        return Filter::LinearMipMapNearest;
      case fastgltf::Filter::NearestMipMapLinear:
        return Filter::NearestMipMapLinear;
      case fastgltf::Filter::LinearMipMapLinear:
        return Filter::LinearMipMapLinear;
      default:
        return Filter::Nearest;
    }
  };

  constexpr auto to_wrap = [](const fastgltf::Wrap wrap) {
    switch (wrap) {
      case fastgltf::Wrap::ClampToEdge:
        return Wrap::ClampToEdge;
      case fastgltf::Wrap::MirroredRepeat:
        return Wrap::MirroredRepeat;
      case fastgltf::Wrap::Repeat:
        return Wrap::Repeat;
      default:
        return Wrap::ClampToEdge;
    }
  };

  for (const auto& sampler: asset->samplers) {
    const auto min_filter = to_filter(sampler.minFilter.value_or(fastgltf::Filter::Nearest));
    const auto mag_filter = to_filter(sampler.magFilter.value_or(fastgltf::Filter::Nearest));
    const auto wrap_s = to_wrap(sampler.wrapS);
    const auto wrap_t = to_wrap(sampler.wrapT);
    result.samplers.push_back(renderer->create_sampler(min_filter, mag_filter, wrap_s, wrap_t));
  }

  for (const auto& texture: asset->textures) {
    const GPUImage image = texture.imageIndex ? result.images.at(*texture.imageIndex)->gpu_image : GPUImage{UINT32_MAX};
    const GPUSampler sampler =
        texture.samplerIndex ? result.samplers.at(*texture.samplerIndex) : GPUSampler{UINT32_MAX};
    result.textures.emplace_back(image, sampler);
  }

  for (const auto& material: asset->materials) {
    MaterialInfo mat_info{};
    mat_info.base_color = glm::make_vec4(material.pbrData.baseColorFactor.data());
    mat_info.emissive_color = glm::make_vec3(material.emissiveFactor.data());
    mat_info.metallic = material.pbrData.metallicFactor;
    mat_info.roughness = material.pbrData.roughnessFactor;

    if (const auto& tex = material.pbrData.baseColorTexture; tex.has_value()) {
      mat_info.textures.albedo.index = static_cast<uint32_t>(tex->textureIndex);
      mat_info.textures.albedo.tex_coord = static_cast<uint32_t>(tex->texCoordIndex);
    }

    if (const auto& tex = material.pbrData.metallicRoughnessTexture; tex.has_value()) {
      mat_info.textures.metal_roughness.index = static_cast<uint32_t>(tex->textureIndex);
      mat_info.textures.metal_roughness.tex_coord = static_cast<uint32_t>(tex->texCoordIndex);
    }

    if (const auto& tex = material.normalTexture; tex.has_value()) {
      mat_info.normal_scale = tex->scale;
      mat_info.textures.normal.index = static_cast<uint32_t>(tex->textureIndex);
      mat_info.textures.normal.tex_coord = static_cast<uint32_t>(tex->texCoordIndex);
    }

    if (const auto& tex = material.occlusionTexture; tex.has_value()) {
      mat_info.occlusion_scale = tex->strength;
      mat_info.textures.occlusion.index = static_cast<uint32_t>(tex->textureIndex);
      mat_info.textures.occlusion.tex_coord = static_cast<uint32_t>(tex->texCoordIndex);
    }

    if (const auto& tex = material.emissiveTexture; tex.has_value()) {
      mat_info.textures.emissive.index = static_cast<uint32_t>(tex->textureIndex);
      mat_info.textures.emissive.tex_coord = static_cast<uint32_t>(tex->texCoordIndex);
    }

    result.materials.push_back(renderer->create_material(mat_info));
  }

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
