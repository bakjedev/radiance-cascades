#pragma once

// #define RENDERER_3D

#ifdef RENDERER_3D
using ActiveRenderer = class Renderer3D;
#else
using ActiveRenderer = class Renderer2D;
#endif
