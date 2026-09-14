#include "input.hpp"

#include "event_dispatcher.hpp"

Input::Input( EventDispatcher& event_dispatcher ) : event_dispatcher_(event_dispatcher) {
    listener_ids_.push_back(event_dispatcher_.listen<KeyDownEvent>(
        [this]( const KeyDownEvent& key ) { on_key_down(key); }));
    listener_ids_.push_back(event_dispatcher_.listen<KeyUpEvent>(
        [this]( const KeyUpEvent& key ) { on_key_up(key); }));
    listener_ids_.push_back(event_dispatcher_.listen<MouseDownEvent>(
        [this]( const MouseDownEvent& button ) { on_mouse_down(button); }));
    listener_ids_.push_back(event_dispatcher_.listen<MouseUpEvent>(
        [this]( const MouseUpEvent& button ) { on_mouse_up(button); }));
    listener_ids_.push_back(event_dispatcher_.listen<MouseMotionEvent>(
        [this]( const MouseMotionEvent& motion ) { on_mouse_motion(motion); }));
    listener_ids_.push_back(event_dispatcher_.listen<MouseWheelEvent>(
        [this]( const MouseWheelEvent& wheel ) { on_mouse_wheel(wheel); }));
}

Input::~Input() {
    for (const auto id : listener_ids_) event_dispatcher_.remove(id);
}

void Input::begin_frame() {
    mouse_delta_x_ = 0.0F;
    mouse_delta_y_ = 0.0F;
    mouse_scroll_ = 0.0F;
}

void Input::end_frame() {
    keys_pressed_ = keys_down_ & ~keys_down_prev_;
    keys_released_ = ~keys_down_ & keys_down_prev_;
    keys_down_prev_ = keys_down_;

    mouse_buttons_pressed_ = mouse_buttons_down_ & ~mouse_buttons_down_prev_;
    mouse_buttons_released_ = ~mouse_buttons_down_ & mouse_buttons_down_prev_;
    mouse_buttons_down_prev_ = mouse_buttons_down_;
}

void Input::on_mouse_motion( const MouseMotionEvent& event ) {
    mouse_x_ = event.x;
    mouse_y_ = event.y;
    mouse_delta_x_ += event.dx;
    mouse_delta_y_ += event.dy;
}
