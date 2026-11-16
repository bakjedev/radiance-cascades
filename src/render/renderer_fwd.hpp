#pragma once

class Renderer2D;
class Renderer3D;

#ifdef RENDERER_3D
using ActiveRenderer = Renderer3D;
#else
using ActiveRenderer = Renderer2D;
#endif
