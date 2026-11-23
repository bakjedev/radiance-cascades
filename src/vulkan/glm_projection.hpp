#pragma once

#include <cmath>
#include <glm/glm.hpp>
#include <optional>

inline glm::mat4 perspective_vk_rev_z(const float y_fov, const float aspect, const float z_near,
                                      const std::optional<float> z_far = std::nullopt)
{
  const float y_scale = 1.0f / std::tan(0.5f * y_fov);
  glm::mat4 matrix(0.0f);
  matrix[0][0] = y_scale / aspect;
  matrix[1][1] = -y_scale;
  matrix[2][3] = -1.0f;
  if (z_far) {
    const float near = z_near, far = *z_far;
    matrix[2][2] = near / (far - near);
    matrix[3][2] = near * far / (far - near);
  } else {
    matrix[2][2] = 0.0f;
    matrix[3][2] = z_near;
  }
  return matrix;
}

inline glm::mat4 ortho_vk_rev_z(const float x_mag, const float y_mag, const float z_near, const float z_far)
{
  const float range = z_far - z_near;
  glm::mat4 matrix(0.0f);
  matrix[0][0] = 1.0f / x_mag;
  matrix[1][1] = -1.0f / y_mag;
  matrix[2][2] = 1.0f / range;
  matrix[3][2] = z_far / range;
  matrix[3][3] = 1.0f;
  return matrix;
}
