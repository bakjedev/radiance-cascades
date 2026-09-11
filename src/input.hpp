#pragma once
#include <bitset>
#include <vector>

#include "input_enums.hpp"

struct KeyDownEvent {
    KeyboardKey key;
};

struct KeyUpEvent {
    KeyboardKey key;
};

struct MouseDownEvent {
    MouseButton button;
};

struct MouseUpEvent {
    MouseButton button;
};

struct MouseMotionEvent {
    float x;
    float y;
    float dx;
    float dy;
};

struct MouseWheelEvent {
    float scroll;
};

class EventDispatcher;

class Input {
public:
    explicit Input(EventDispatcher &event_dispatcher);

    ~Input();

    Input(const Input &) = delete;

    Input &operator=(const Input &) = delete;

    Input(Input &&) = delete;

    Input &operator=(Input &&) = delete;

    void begin_frame();

    void end_frame();

    [[nodiscard]] bool key_down(const KeyboardKey key) const { return test(keys_down_, key); }
    [[nodiscard]] bool key_pressed(const KeyboardKey key) const { return test(keys_pressed_, key); }
    [[nodiscard]] bool key_released(const KeyboardKey key) const { return test(keys_released_, key); }

    [[nodiscard]] bool mouse_down(const MouseButton button) const { return test(mouse_buttons_down_, button); }
    [[nodiscard]] bool mouse_pressed(const MouseButton button) const { return test(mouse_buttons_pressed_, button); }
    [[nodiscard]] bool mouse_released(const MouseButton button) const { return test(mouse_buttons_released_, button); }

    [[nodiscard]] float mouse_x() const { return mouse_x_; }
    [[nodiscard]] float mouse_y() const { return mouse_y_; }
    [[nodiscard]] float mouse_delta_x() const { return mouse_delta_x_; }
    [[nodiscard]] float mouse_delta_y() const { return mouse_delta_y_; }
    [[nodiscard]] float mouse_scroll() const { return mouse_scroll_; }

private:
    template<size_t N, class T>
    static bool test(const std::bitset<N> &bits, T input) {
        const auto i = static_cast<size_t>(input);
        return i < N && bits.test(i);
    }

    template<size_t N, class T>
    static void set(std::bitset<N> &bits, T input, bool value) {
        const auto i = static_cast<size_t>(input);
        if (i < N) bits.set(i, value);
    }

    void on_key_down(const KeyDownEvent &event) { set(keys_down_, event.key, true); }
    void on_key_up(const KeyUpEvent &event) { set(keys_down_, event.key, false); }
    void on_mouse_down(const MouseDownEvent &event) { set(mouse_buttons_down_, event.button, true); }
    void on_mouse_up(const MouseUpEvent &event) { set(mouse_buttons_down_, event.button, false); }

    void on_mouse_motion(const MouseMotionEvent &event);

    void on_mouse_wheel(const MouseWheelEvent &event) { mouse_scroll_ += event.scroll; }

    std::bitset<static_cast<size_t>(KeyboardKey::Count)> keys_down_;
    std::bitset<static_cast<size_t>(KeyboardKey::Count)> keys_down_prev_;
    std::bitset<static_cast<size_t>(KeyboardKey::Count)> keys_pressed_;
    std::bitset<static_cast<size_t>(KeyboardKey::Count)> keys_released_;

    std::bitset<static_cast<size_t>(MouseButton::Count)> mouse_buttons_down_;
    std::bitset<static_cast<size_t>(MouseButton::Count)> mouse_buttons_down_prev_;
    std::bitset<static_cast<size_t>(MouseButton::Count)> mouse_buttons_pressed_;
    std::bitset<static_cast<size_t>(MouseButton::Count)> mouse_buttons_released_;

    float mouse_x_{0.0F};
    float mouse_y_{0.0F};
    float mouse_delta_x_{0.0F};
    float mouse_delta_y_{0.0F};
    float mouse_scroll_{0.0F};

    EventDispatcher &event_dispatcher_;
    std::vector<uint64_t> listener_ids_;
};
