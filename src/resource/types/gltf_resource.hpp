#pragma once
#include <glm/glm.hpp>
#include <string>
#include <vector>
#include "src/render/renderer_fwd.hpp"
#include "src/render/renderer_types.hpp"
#include "src/resource/resource_manager.hpp"

class FileSystem;

struct GLTFResource {
  struct Node {
    std::string name;
    glm::mat4 transform;

    uint32_t mesh_id;
    uint32_t light_id;
    uint32_t camera_id;
  };

  struct Primitive {
    GeometryRange range;
    GPUMaterial material;
  };

  struct Mesh {
    std::vector<Primitive> primitives;
  };

  struct Light {
    enum class Type : uint8_t {
      Directional,
      Spot,
      Point,
    };

    Type type;
    glm::vec3 color;
    float intensity;

    float range;
    float inner_cone_angle;
    float outer_cone_angle;
  };

  struct Camera {
    glm::mat4 projection;
  };

  std::vector<Node> nodes;
  std::vector<Mesh> meshes;
  std::vector<Light> lights;
  std::vector<Camera> cameras;
  std::vector<ResourceHandle<ImageResource>> images;
  std::vector<GPUSampler> samplers;
  std::vector<std::pair<GPUImage, GPUSampler>> textures;
  std::vector<GPUMaterial> materials;
};


struct GLTFResourceLoader {
  GLTFResource operator()(const std::string& path) const;

  FileSystem* file_system;
  ResourceManager* resource_manager;
  Renderer3D* renderer;
};
