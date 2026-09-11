#pragma once
#include <functional>
#include <typeindex>
#include <framework/util/flat_hash_map.hpp>


// copied/inspired from KyrietS/tinyevents
class EventDispatcher {
    using ListenerFunc = std::function<void(const void *)>;
    using Listeners = fwrk::flat_hash_map<uint64_t, ListenerFunc>;

public:
    template<class T>
    uint64_t listen(std::function<void(const T &)> func) {
        auto &listeners = listeners_by_type_[typeid(T)];
        const auto listener_id = next_listener_id_++;

        listeners[listener_id] = [func](const void *ptr) {
            func(*static_cast<const T *>(ptr));
        };
        return listener_id;
    }

    template<class T>
    void dispatch(const T &event) {
        auto iter = listeners_by_type_.find(typeid(T));
        if (iter == listeners_by_type_.end()) return;

        const auto &[type, listeners] = *iter;

        // to allow for deletion mid iteration
        std::vector<ListenerFunc> funcs;
        funcs.reserve(listeners.size());
        for (auto &[_, func]: listeners) {
            funcs.push_back(func);
        }

        for (auto &func: funcs) { func(&event); }
    }

    void remove(const uint64_t id) {
        for (auto &[_, listeners]: listeners_by_type_) {
            listeners.erase(listeners.find(id));
        }
    }

private:
    fwrk::flat_hash_map<std::type_index, Listeners> listeners_by_type_;
    uint64_t next_listener_id_ = 0;
};
