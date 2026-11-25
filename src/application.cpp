#include "application.hpp"
#include <SDL3/SDL_vulkan.h>
#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/matrix_transform.hpp>
#include "camera_fly.hpp"
#include "resource/types/gltf_resource.hpp"
#include "resource/types/image_resource.hpp"
#include "resource/types/shader_resource.hpp"
#include "vulkan/glm_projection.hpp"

struct QuitEvent {};

Application::Application(const uint32_t window_width, const uint32_t window_height) :
    window_(event_dispatcher_, window_width, window_height, "everything.. seems to be in order."),
    input_(event_dispatcher_), user_interface_(window_.get()), renderer_(create_renderer()), scene_(create_scene())
{
}

Application::~Application() = default;

void Application::run()
{
  event_dispatcher_.listen<QuitEvent>([this](const QuitEvent&) { running = false; });

  file_system_.add_route("assets", "../assets");

#ifdef RENDERER_3D
  CameraFly camera_fly{};
  if (const auto camera = scene_.active_camera()) {
    camera_fly.set_camera(scene_, *camera);
  }
#else
  auto texture_resource = resource_manager_.create_from_file<ImageResource>(
      "image.png", ImageResourceLoader{&file_system_, &renderer_}, ImageColorSpace::sRGB);
#endif

  using clock = std::chrono::high_resolution_clock;
  auto last_time = clock::now();

  while (running) {
    const auto current_time = clock::now();
    [[maybe_unused]] const float delta_time = std::chrono::duration<float>(current_time - last_time).count();
    last_time = current_time;

    input_.begin_frame();
    poll_events();
    input_.end_frame();

    if (input_.key_down(KeyboardKey::Escape)) {
      running = false;
    }

    glm::mat4 projection, view;
#ifdef RENDERER_3D
    for (const auto& [transform, mesh_id, _, light_id]: scene_.entities()) {
      if (mesh_id != UINT32_MAX) {
        for (const auto& [range, material]: scene_.resource()->meshes.at(mesh_id).primitives) {
          renderer_.submit_geometry({range, material, transform});
        }
      }

      if (light_id != UINT32_MAX) {
        std::visit(
            [&]<typename T>(const T& light) {
              const auto position = glm::vec3(transform[3]);
              if constexpr (std::is_same_v<T, Scene::PointLight>) {
                renderer_.submit_light(PointLightSubmission{position, light.color * light.intensity, light.range});
              } else if constexpr (std::is_same_v<T, Scene::SpotLight>) {
                renderer_.submit_light(SpotLightSubmission{{position, light.color * light.intensity, light.range},
                                                           -glm::vec3(transform[2]),
                                                           std::cos(light.inner_cone),
                                                           std::cos(light.outer_cone)});
              }
            },
            scene_.lights().at(light_id));
      }
    }

    if (auto active_camera = scene_.active_camera()) {
      camera_fly.update(scene_, input_, delta_time);

      const auto& cam_entity = scene_.entities().at(*active_camera);
      view = glm::inverse(cam_entity.transform);

      const auto& camera = scene_.cameras().at(cam_entity.camera_id);
      const float aspect = static_cast<float>(window_.width()) / static_cast<float>(window_.height());

      if (auto* perspective = std::get_if<Scene::PerspectiveCamera>(&camera)) {
        projection = perspective_vk_rev_z(perspective->y_fov, aspect, perspective->z_near, perspective->z_far);
      } else if (auto* orthographic = std::get_if<Scene::OrthoGraphicCamera>(&camera)) {
        projection =
            ortho_vk_rev_z(orthographic->x_mag, orthographic->y_mag, orthographic->z_near, *orthographic->z_far);
      }
    }
#else
    if (input_.mouse_down(MouseButton::Left) && !UserInterface::consuming()) {
      renderer_.plot({input_.mouse_x(), input_.mouse_y()});
    }

    renderer_.draw_sprite(Renderer2D::Sprite{.texture = texture_resource->gpu_image,
                                             .pos = {100.0f, 500.0f},
                                             .size = {1.0f, 1.0f},
                                             .tint = {1.0f, 1.0f, 1.0f}});
#endif

    auto* imgui_draw_data = user_interface_.render(renderer_);

    renderer_.render({.window_size = {window_.width(), window_.height()},
                      .imgui_draw_data = imgui_draw_data,
                      .view = view,
                      .projection = projection});
  }
}

void Application::poll_events()
{
  SDL_Event event;
  while (SDL_PollEvent(&event)) {
    UserInterface::process_event(event);
    switch (event.type) {
      case SDL_EVENT_QUIT:
        event_dispatcher_.dispatch(QuitEvent{});
        break;
      case SDL_EVENT_KEY_DOWN:
        event_dispatcher_.dispatch(KeyDownEvent{static_cast<KeyboardKey>(event.key.scancode)});
        break;
      case SDL_EVENT_KEY_UP:
        event_dispatcher_.dispatch(KeyUpEvent{static_cast<KeyboardKey>(event.key.scancode)});
        break;
      case SDL_EVENT_MOUSE_BUTTON_DOWN:
        event_dispatcher_.dispatch(MouseDownEvent{static_cast<MouseButton>(event.button.button)});
        break;
      case SDL_EVENT_MOUSE_BUTTON_UP:
        event_dispatcher_.dispatch(MouseUpEvent{static_cast<MouseButton>(event.button.button)});
        break;
      case SDL_EVENT_MOUSE_MOTION:
        event_dispatcher_.dispatch(
            MouseMotionEvent{event.motion.x, event.motion.y, event.motion.xrel, event.motion.yrel});
        break;
      case SDL_EVENT_MOUSE_WHEEL:
        event_dispatcher_.dispatch(MouseWheelEvent{static_cast<float>(event.wheel.integer_y)});
        break;
      case SDL_EVENT_WINDOW_RESIZED:
        event_dispatcher_.dispatch(WindowResizeEvent{event.window.data1, event.window.data2});
        break;
      default:
        break;
    }
  }
}

ActiveRenderer Application::create_renderer()
{
  uint32_t sdl_instance_extension_count = 0;
  const auto* const sdl_instance_extensions = SDL_Vulkan_GetInstanceExtensions(&sdl_instance_extension_count);

  return ActiveRenderer(RendererCreateInfo{
      .instance_extensions = {sdl_instance_extensions, sdl_instance_extension_count},
      .create_surface = [this](vk::Instance instance) -> vk::SurfaceKHR {
        VkSurfaceKHR surface = VK_NULL_HANDLE;
        if (!SDL_Vulkan_CreateSurface(window_.get(), instance, nullptr, &surface)) {
          throw std::runtime_error("Failed to create Vulkan Surface with SDL");
        }
        return surface;
      },
      .initial_extent = {window_.width(), window_.height()},
      .load_shader =
          [this](const std::string& name) {
            auto shader_resource =
                resource_manager_.create_from_file<ShaderResource>(name, ShaderResourceLoader{&file_system_});
            return shader_resource->code;
          }});
}

Scene Application::create_scene()
{
#ifdef RENDERER_3D
  const auto gltf_resource = resource_manager_.create_from_file<GLTFResource>(
      "HelmetWithLight.glb", GLTFResourceLoader{&file_system_, &resource_manager_, &renderer_});
  return Scene{gltf_resource};
#else
  return Scene{{}};
#endif
}
