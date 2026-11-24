#pragma once
#include <cstdint>
#include "input.hpp"
#include "scene.hpp"

struct CameraFly {
  uint32_t entity = UINT32_MAX;
  float yaw = 0.0f, pitch = 0.0f;
  float speed = 5.0f, sensitivity = 0.002f;

  void set_camera(Scene& scene, const uint32_t id)
  {
    entity = id;
    const glm::vec3 forward =
        glm::normalize(glm::vec3(scene.entities().at(entity).transform * glm::vec4(0.0f, 0.0f, -1.0f, 0.0f)));
    yaw = std::atan2(-forward.x, -forward.z);
    pitch = std::asin(glm::clamp(forward.y, -1.0f, 1.0f));
  }

  void update(Scene& scene, const Input& input, float delta_time)
  {
    if (entity == UINT32_MAX) return;
    if (!input.mouse_down(MouseButton::Right)) return;
    glm::mat4& transform = scene.entities().at(entity).transform;
    auto pos = glm::vec3(transform[3]);

    yaw -= input.mouse_delta_x() * sensitivity;
    pitch -= input.mouse_delta_y() * sensitivity;
    pitch = glm::clamp(pitch, glm::radians(-89.0f), glm::radians(89.0f));

    glm::vec3 forward{
        -sinf(yaw) * cosf(pitch),
        sinf(pitch),
        -cosf(yaw) * cosf(pitch),
    };

    constexpr auto world_up = glm::vec3(0.0f, 1.0f, 0.0f);
    const glm::vec3 right = glm::normalize(glm::cross(forward, world_up));
    const glm::vec3 up = glm::cross(right, forward);

    glm::vec3 move{};
    if (input.key_down(KeyboardKey::W)) move += forward;
    if (input.key_down(KeyboardKey::S)) move -= forward;
    if (input.key_down(KeyboardKey::D)) move += right;
    if (input.key_down(KeyboardKey::A)) move -= right;
    if (input.key_down(KeyboardKey::Space)) move += world_up;
    if (input.key_down(KeyboardKey::LeftControl)) move -= world_up;

    if (glm::dot(move, move) > 0.0f) {
      float final_speed = speed * (input.key_down(KeyboardKey::LeftShift) ? 4.0f : 1.0f);
      pos += glm::normalize(move) * final_speed * delta_time;
    }

    transform[0] = glm::vec4(right, 0.0f);
    transform[1] = glm::vec4(up, 0.0f);
    transform[2] = glm::vec4(-forward, 0.0f);
    transform[3] = glm::vec4(pos, 1.0f);
  }
};
