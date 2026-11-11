#pragma once
#include "renderer_2d.hpp"
#include "renderer_3d.hpp"
static_assert(Renderer<Renderer2D>);
static_assert(Renderer<Renderer3D>);
using CurrentRenderer = Renderer3D;
