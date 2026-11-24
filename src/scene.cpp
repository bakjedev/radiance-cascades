#include "scene.hpp"

#include <glm/ext/matrix_transform.hpp>

#include "resource/types/gltf_resource.hpp"
#include "resource/types/image_resource.hpp"

Scene::Scene(const ResourceHandle<GLTFResource>& resource) : scene_resource_(resource)
{
  if (!resource) return;

  for (const auto& node: resource->nodes) {
    if (node.camera_id != UINT32_MAX) {
      const auto& [type, y_fov, x_mag, y_mag, z_near, z_far] = resource->cameras.at(node.camera_id);
      cameras_.emplace_back(
          type == GLTFResource::Camera::Type::Perspective ? Camera::Type::Perspective : Camera::Type::Orthographic,
          y_fov, x_mag, y_mag, z_near, z_far);
    }
    entities_.emplace_back(node.transform, node.mesh_id, node.camera_id);
  }

  active_camera_ = UINT32_MAX;
  for (uint32_t i = 0; i < entities_.size(); ++i) {
    if (entities_[i].camera_id != UINT32_MAX) {
      active_camera_ = i;
      break;
    }
  }
  if (active_camera_ == UINT32_MAX) {
    constexpr Camera cam{.type = Camera::Type::Perspective,
                         .y_fov = glm::radians(85.0f),
                         .x_mag = 0.0f,
                         .y_mag = 0.0f,
                         .z_near = 0.1f,
                         .z_far = std::nullopt};

    cameras_.push_back(cam);
    entities_.push_back(Entity{.transform = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, 3.0f)),
                               .mesh_id = UINT32_MAX,
                               .camera_id = static_cast<uint32_t>(cameras_.size() - 1)});
    active_camera_ = static_cast<uint32_t>(entities_.size() - 1);
  }
}
