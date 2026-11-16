#pragma once

// #define RENDERER_3D

#ifdef RENDERER_3D
#include "src/render/renderer_3d.hpp"
#else
#include "src/render/renderer_2d.hpp"
#endif
