#include "application.hpp"
#include <SDL3/SDL_vulkan.h>
#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <iostream>

#include "resource/types/gltf_resource.hpp"
#include "resource/types/image_resource.hpp"
#include "resource/types/shader_resource.hpp"

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

#ifndef RENDERER_3D
  auto texture_resource = resource_manager_.create_from_file<ImageResource>(
      "image.png", ImageResourceLoader{&file_system_, &renderer_}, ImageColorSpace::sRGB);
#endif

  while (running) {
    input_.begin_frame();
    poll_events();
    input_.end_frame();

    if (input_.key_down(KeyboardKey::Escape)) {
      running = false;
    }

    glm::mat4 projection, view;
#ifdef RENDERER_3D
    for (const auto& [transform, mesh_id, camera_id]: scene_.entities()) {
      for (const auto& [range, material]: scene_.resource()->meshes.at(mesh_id).primitives) {
        renderer_.draw_geometry({range, material, transform});
      }
    }
    if (scene_.active_camera() != UINT32_MAX) {
      const auto& camera_entity = scene_.entities().at(scene_.active_camera());
      const auto& camera = scene_.resource()->cameras.at(camera_entity.camera_id);
      projection = camera.projection;
      view = camera_entity.transform;
    } else {
      projection =
          glm::perspective(glm::radians(90.0f),
                           static_cast<float>(window_.width()) / static_cast<float>(window_.height()), 100.0f, 0.1f);
      // projection[1][1] *= -1.0f;
      view = glm::lookAt(glm::vec3(0.0f, 5.0f, -5.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
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
      "DamagedHelmet.glb", GLTFResourceLoader{&file_system_, &resource_manager_, &renderer_});
  return Scene{gltf_resource};
#else
  return Scene{{}};
#endif
}
