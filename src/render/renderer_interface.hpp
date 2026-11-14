#pragma once
#include "framework/graph.hpp"
#include "renderer_types.hpp"

class RenderBackend;

template<class T>
concept RendererInterface =
    std::constructible_from<T, const RendererCreateInfo&> && std::derived_from<T, RenderBackend> &&
    requires(T renderer, RenderInfo info) { renderer.render(info); };
