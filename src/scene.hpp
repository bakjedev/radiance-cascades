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

  struct Camera {
    enum class Type : uint8_t { Perspective, Orthographic };
    Type type;
    float y_fov;
    float x_mag, y_mag;
    float z_near;
    std::optional<float> z_far;
  };

  explicit Scene(const ResourceHandle<GLTFResource>& resource);

  [[nodiscard]] auto& entities() { return entities_; }
  [[nodiscard]] auto& cameras() { return cameras_; }
  [[nodiscard]] const auto& resource() const { return scene_resource_; }
  [[nodiscard]] auto active_camera()
  {
    return active_camera_ != UINT32_MAX ? std::optional(active_camera_) : std::nullopt;
  }

private:
  ResourceHandle<GLTFResource> scene_resource_;
  std::vector<Entity> entities_;

  std::vector<Camera> cameras_;
  uint32_t active_camera_;
};
