#pragma once

#include <filesystem>
#include <tuple>

#include "resource.hpp"

template<typename... Ts>
class ResourceManager {
public:
    template<typename T>
    ResourceStorage<T>& get_storage() {
        return std::get<ResourceStorage<T>>(storages_);
    }

    template<typename T>
    const ResourceStorage<T>& get_storage() const {
        return std::get<ResourceStorage<T>>(storages_);
    }

    template<typename T, typename Loader, typename... Args>
    ResourceHandle<T> load( const std::string& key, Loader&& loader, Args&&... args ) {
        return get_storage<T>().load(key, std::forward<Loader>(loader), std::forward<Args>(args)...);
    }

    template<typename T, typename Loader, typename... Args>
    ResourceHandle<T> create_from_file( const std::string& key, Loader&& loader, Args&&... args ) {
        return get_storage<T>().load(key, std::forward<Loader>(loader), key, std::forward<Args>(args)...);
    }

private:
    std::tuple<ResourceStorage<Ts>...> storages_;
};
