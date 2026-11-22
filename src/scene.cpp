#include "scene.hpp"
#include "resource/types/image_resource.hpp"

Scene::Scene(const ResourceHandle<GLTFResource>& resource) : scene_resource_(resource)
{
  if (!resource) return;

  active_camera_ = UINT32_MAX;
  for (const auto& node: resource->nodes) {
    if (node.camera_id != UINT32_MAX && active_camera_ == UINT32_MAX) {
      active_camera_ = static_cast<uint32_t>(entities_.size());
    }
    entities_.emplace_back(node.transform, node.mesh_id, node.camera_id);
  }
}
