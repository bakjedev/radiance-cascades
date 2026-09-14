#pragma once

class Window;
class EventDispatcher;

class Renderer2D {
public:
    Renderer2D(Window &window, EventDispatcher &event_dispatcher);

private:
    Window &window_;
    EventDispatcher &event_dispatcher_;
};
