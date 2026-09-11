#pragma once
#include <stdexcept>

#include "SDL3/SDL.h"

class SDLInit {
public:
    SDLInit() {
        if (!SDL_Init(SDL_INIT_VIDEO)) {
            throw std::runtime_error(std::string("SDL_Init: ") + SDL_GetError());
        }
    }

    ~SDLInit() { SDL_Quit(); }
};
