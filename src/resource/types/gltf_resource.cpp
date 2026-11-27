#include "gltf_resource.hpp"
#include <cmath>
#include <fastgltf/core.hpp>
#include <fastgltf/glm_element_traits.hpp>
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
  const auto full_path = std::filesystem::weakly_canonical(full_path_str.value);

  auto data = fastgltf::GltfDataBuffer::FromPath(full_path);
  if (data.error() != fastgltf::Error::None) {
    throw std::runtime_error("The glTF file couldn't be loaded, or the buffer could not be allocated.");
  }

  fastgltf::Parser parser{fastgltf::Extensions::KHR_lights_punctual};

  auto asset = parser.loadGltf(data.get(), full_path.parent_path(), fastgltf::Options::LoadExternalBuffers);
  if (const auto error = asset.error(); error != fastgltf::Error::None) {
    throw std::runtime_error(std::string(fastgltf::getErrorName(asset.error())) + ": " +
                             std::string(fastgltf::getErrorMessage(asset.error())));
  }

  std::vector image_is_srgb(asset->images.size(), false);
  auto set_srgb = [&](const auto& texture_info) {
    if (!texture_info) return;
    if (auto img = asset->textures[texture_info->textureIndex].imageIndex) image_is_srgb[*img] = true;
  };
  for (const auto& material: asset->materials) {
    set_srgb(material.pbrData.baseColorTexture);
    set_srgb(material.emissiveTexture);
  }

  for (size_t i = 0; i < asset->images.size(); ++i) {
    const auto color_space = image_is_srgb[i] ? ImageColorSpace::sRGB : ImageColorSpace::Linear;
    std::visit(fastgltf::visitor{
                   [&]([[maybe_unused]] const auto& arg) { result.images.emplace_back(); },
                   [&](const fastgltf::sources::URI& uri) {
                     const auto resource = resource_manager->create_from_file<ImageResource>(
                         full_path.parent_path().string() + "/" + std::string(uri.uri->path()),
                         ImageResourceLoader{file_system, renderer}, color_space);
                     result.images.push_back(resource);
                   },
                   [&](const fastgltf::sources::Array& array) {
                     const auto resource = resource_manager->load<ImageResource>(
                         std::format("image_{}_{}", full_path.string(), i), ImageResourceLoader{file_system, renderer},
                         array.bytes, color_space);
                     result.images.push_back(resource);
                   },
                   [&](const fastgltf::sources::Vector& vector) {
                     const auto resource = resource_manager->load<ImageResource>(
                         std::format("image_{}_{}", full_path.string(), i), ImageResourceLoader{file_system, renderer},
                         vector.bytes, color_space);
                     result.images.push_back(resource);
                   },
                   [&](const fastgltf::sources::BufferView& buffer_view) {
                     const auto& view = asset->bufferViews[buffer_view.bufferViewIndex];
                     const auto& buffer = asset->buffers[view.bufferIndex];
                     std::visit(
                         fastgltf::visitor{
                             [&]([[maybe_unused]] const auto& arg) { result.images.emplace_back(); },
                             [&](const fastgltf::sources::Array& array) {
                               const auto resource = resource_manager->load<ImageResource>(
                                   std::format("image_{}_{}", full_path.string(), i),
                                   ImageResourceLoader{file_system, renderer},
                                   std::span(array.bytes).subspan(view.byteOffset, view.byteLength), color_space);
                               result.images.push_back(resource);
                             },
                             [&](const fastgltf::sources::Vector& vector) {
                               const auto resource = resource_manager->load<ImageResource>(
                                   std::format("image_{}_{}", full_path.string(), i),
                                   ImageResourceLoader{file_system, renderer},
                                   std::span(vector.bytes).subspan(view.byteOffset, view.byteLength), color_space);
                               result.images.push_back(resource);
                             }},
                         buffer.data);
                   },
               },
               asset->images[i].data);
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
    const auto min_filter = to_filter(sampler.minFilter.value_or(fastgltf::Filter::LinearMipMapLinear));
    const auto mag_filter = to_filter(sampler.magFilter.value_or(fastgltf::Filter::Linear));
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
    mat_info.alpha_mode = static_cast<uint32_t>(material.alphaMode);
    mat_info.alpha_cutoff = material.alphaCutoff;

    const auto fill_texture = [&](auto& texture, const auto& info) {
      const auto& [image, sampler] = result.textures[info.textureIndex];
      texture.index = image.id;
      texture.tex_coord = static_cast<uint32_t>(info.texCoordIndex);
      texture.sampler = sampler.id;
    };

    if (const auto& tex_info = material.pbrData.baseColorTexture) fill_texture(mat_info.textures.albedo, *tex_info);
    if (const auto& tex_info = material.pbrData.metallicRoughnessTexture)
      fill_texture(mat_info.textures.metal_roughness, *tex_info);
    if (const auto& tex_info = material.normalTexture) {
      mat_info.normal_scale = tex_info->scale;
      fill_texture(mat_info.textures.normal, *tex_info);
    }
    if (const auto& tex_info = material.occlusionTexture) {
      mat_info.occlusion_scale = tex_info->strength;
      fill_texture(mat_info.textures.occlusion, *tex_info);
    }
    if (const auto& tex_info = material.emissiveTexture) fill_texture(mat_info.textures.emissive, *tex_info);

    result.materials.push_back(renderer->create_material(mat_info));
  }

  std::vector<uint32_t> indices;
  std::vector<Vertex> vertices;
  for (const auto& mesh: asset->meshes) {
    auto& [primitives] = result.meshes.emplace_back();
    primitives.reserve(mesh.primitives.size());

    for (const auto& primitive: mesh.primitives) {
      GeometryRange range{};
      range.first_index = static_cast<uint32_t>(indices.size());

      const auto start_vertex = static_cast<uint32_t>(vertices.size());
      range.vertex_offset = static_cast<int32_t>(start_vertex);

      {
        fastgltf::Accessor& index_accessor = asset->accessors[primitive.indicesAccessor.value()];
        range.index_count = static_cast<uint32_t>(index_accessor.count);
        indices.reserve(indices.size() + index_accessor.count);
        fastgltf::iterateAccessor<uint32_t>(asset.get(), index_accessor,
                                            [&](const uint32_t idx) { indices.push_back(idx); });
      }

      {
        fastgltf::Accessor& pos_accessor = asset->accessors[primitive.findAttribute("POSITION")->accessorIndex];
        vertices.resize(vertices.size() + pos_accessor.count);

        fastgltf::iterateAccessorWithIndex<glm::vec3>(asset.get(), pos_accessor,
                                                      [&](const glm::vec3 pos, const size_t idx) {
                                                        vertices[start_vertex + idx] = Vertex{.position = pos,
                                                                                              .normal = {1, 0, 0},
                                                                                              .tex_coord_0 = {0, 0},
                                                                                              .tex_coord_1 = {0, 0},
                                                                                              .tangent = {1, 0, 0, 1}};
                                                      });
      }

      auto normals = primitive.findAttribute("NORMAL");
      if (normals != primitive.attributes.end()) {
        fastgltf::iterateAccessorWithIndex<glm::vec3>(
            asset.get(), asset->accessors[normals->accessorIndex],
            [&](const glm::vec3 normal, const size_t idx) { vertices[start_vertex + idx].normal = normal; });
      }

      auto uvs_0 = primitive.findAttribute("TEXCOORD_0");
      auto uvs_1 = primitive.findAttribute("TEXCOORD_1");
      if (uvs_0 != primitive.attributes.end()) {
        fastgltf::iterateAccessorWithIndex<glm::vec2>(
            asset.get(), asset->accessors[uvs_0->accessorIndex],
            [&](const glm::vec2 uv, const size_t idx) { vertices[start_vertex + idx].tex_coord_0 = uv; });
      }
      if (uvs_1 != primitive.attributes.end()) {
        fastgltf::iterateAccessorWithIndex<glm::vec2>(
            asset.get(), asset->accessors[uvs_1->accessorIndex],
            [&](const glm::vec2 uv, const size_t idx) { vertices[start_vertex + idx].tex_coord_1 = uv; });
      }

      auto tangents = primitive.findAttribute("TANGENT");
      if (tangents != primitive.attributes.end()) {
        fastgltf::iterateAccessorWithIndex<glm::vec4>(
            asset.get(), asset->accessors[tangents->accessorIndex],
            [&](const glm::vec4 tangent, const size_t idx) { vertices[start_vertex + idx].tangent = tangent; });
      }

      const GPUMaterial material =
          primitive.materialIndex ? result.materials.at(*primitive.materialIndex) : GPUMaterial{UINT32_MAX};

      primitives.emplace_back(range, material);
    }
  }
  renderer->upload_vertices(vertices, indices);

  for (const auto& light: asset->lights) {
    result.lights.emplace_back(static_cast<GLTFResource::Light::Type>(light.type), glm::make_vec3(light.color.data()),
                               light.intensity, light.range.value_or(0.0f), light.innerConeAngle.value_or(0.0f),
                               light.outerConeAngle.value_or(0.0f));
  }

  for (const auto& camera: asset->cameras) {
    std::visit(fastgltf::visitor{
                   [&](const fastgltf::Camera::Orthographic& orthographic) {
                     result.cameras.emplace_back(GLTFResource::Camera::Type::Orthographic, 0.0f, orthographic.xmag,
                                                 orthographic.ymag, orthographic.znear, orthographic.zfar);
                   },
                   [&](const fastgltf::Camera::Perspective& perspective) {
                     result.cameras.emplace_back(GLTFResource::Camera::Type::Perspective, perspective.yfov, 0.0f, 0.0f,
                                                 perspective.znear, perspective.zfar);
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
