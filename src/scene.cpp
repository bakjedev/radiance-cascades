#include "scene.hpp"

#include <glm/ext/matrix_transform.hpp>

#include "resource/types/gltf_resource.hpp"
#include "resource/types/image_resource.hpp"

Scene::Scene(const ResourceHandle<GLTFResource>& resource) : scene_resource_(resource)
{
  if (!resource) return;

  for (const auto& node: resource->nodes) {
    uint32_t camera_id = UINT32_MAX;
    if (node.camera_id != UINT32_MAX) {
      camera_id = static_cast<uint32_t>(cameras_.size());
      const auto& [type, y_fov, x_mag, y_mag, z_near, z_far] = resource->cameras.at(node.camera_id);
      switch (type) {
        case GLTFResource::Camera::Type::Perspective:
          cameras_.emplace_back(PerspectiveCamera{{z_near, z_far}, y_fov});
          break;
        case GLTFResource::Camera::Type::Orthographic:
          cameras_.emplace_back(OrthoGraphicCamera{{z_near, z_far}, x_mag, y_mag});
          break;
      }
    }

    uint32_t light_id = UINT32_MAX;
    if (node.light_id != UINT32_MAX) {
      const auto& [type, color, intensity, range, inner_cone_angle, outer_cone_angle] =
          resource->lights.at(node.light_id);
      switch (type) {
        case GLTFResource::Light::Type::Point:
          light_id = static_cast<uint32_t>(lights_.size());
          lights_.emplace_back(PointLight{color, intensity, range});
          break;
        case GLTFResource::Light::Type::Spot:
          light_id = static_cast<uint32_t>(lights_.size());
          lights_.emplace_back(SpotLight{{color, intensity, range}, inner_cone_angle, outer_cone_angle});
          break;
        default:
          break;
      }
    }

    entities_.emplace_back(node.transform, node.mesh_id, camera_id, light_id, true);
  }

  active_camera_ = UINT32_MAX;
  for (uint32_t i = 0; i < entities_.size(); ++i) {
    if (entities_[i].camera_id != UINT32_MAX) {
      active_camera_ = i;
      break;
    }
  }
  if (active_camera_ == UINT32_MAX) {
    cameras_.emplace_back(PerspectiveCamera{{0.1f, std::nullopt}, glm::radians(85.0f)});
    entities_.push_back(Entity{
        .transform = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, 3.0f)),
        .mesh_id = UINT32_MAX,
        .camera_id = static_cast<uint32_t>(cameras_.size() - 1),
        .light_id = UINT32_MAX,
        .cast_shadow = false,
    });
    active_camera_ = static_cast<uint32_t>(entities_.size() - 1);
  }
}
