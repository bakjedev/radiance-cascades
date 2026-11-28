#pragma once
#include <variant>


#include "resource/resource.hpp"
#include "resource/types/gltf_resource.hpp"

class Scene {
public:
  struct Entity {
    glm::mat4 transform;
    uint32_t mesh_id;
    uint32_t camera_id;
    uint32_t light_id;
    bool cast_shadow;
  };

  struct CameraBase {
    float z_near;
    std::optional<float> z_far;
  };
  struct PerspectiveCamera : CameraBase {
    float y_fov;
  };
  struct OrthoGraphicCamera : CameraBase {
    float x_mag, y_mag;
  };
  using Camera = std::variant<PerspectiveCamera, OrthoGraphicCamera>;

  struct LightBase {
    glm::vec3 color;
    float intensity;
    float range;
  };
  struct PointLight : LightBase {};
  struct SpotLight : LightBase {
    float inner_cone;
    float outer_cone;
  };
  using Light = std::variant<PointLight, SpotLight>;

  explicit Scene(const ResourceHandle<GLTFResource>& resource);

  [[nodiscard]] auto& entities() { return entities_; }
  [[nodiscard]] auto& cameras() { return cameras_; }
  [[nodiscard]] auto& lights() { return lights_; }
  [[nodiscard]] const auto& resource() const { return scene_resource_; }
  [[nodiscard]] auto active_camera()
  {
    return active_camera_ != UINT32_MAX ? std::optional(active_camera_) : std::nullopt;
  }

private:
  ResourceHandle<GLTFResource> scene_resource_;
  std::vector<Entity> entities_;

  std::vector<Camera> cameras_;
  std::vector<Light> lights_;
  uint32_t active_camera_;
};
