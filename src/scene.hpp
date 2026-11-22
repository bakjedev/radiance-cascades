#pragma once
#include "resource/resource.hpp"
#include "resource/types/gltf_resource.hpp"

class Scene {
public:
  struct Entity {
    glm::mat4 transform;
    uint32_t mesh_id;
    uint32_t camera_id;
  };

  explicit Scene(const ResourceHandle<GLTFResource>& resource);

  [[nodiscard]] const auto& entities() const { return entities_; }
  [[nodiscard]] const auto& resource() const { return scene_resource_; }
  [[nodiscard]] auto active_camera() const { return active_camera_; }

private:
  ResourceHandle<GLTFResource> scene_resource_;
  std::vector<Entity> entities_;
  uint32_t active_camera_;
};
