#pragma once
#include <cassert>
#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "resource_callback.hpp"

template<typename T>
class ResourceStorage;

template<typename T>
class ResourceHandle
{
public:
  ResourceHandle() = default;
  ~ResourceHandle() { release(); }

  ResourceHandle(const ResourceHandle& other) : index_(other.index_), storage_(other.storage_)
  {
    if (valid())
    {
      storage_->acquire(index_);
    }
  }

  ResourceHandle(ResourceHandle&& other) noexcept : index_(other.index_), storage_(other.storage_)
  {
    other.storage_ = nullptr;
  }

  ResourceHandle& operator=(const ResourceHandle& other)
  {
    if (this != &other)
    {
      release();
      index_ = other.index_;
      storage_ = other.storage_;
      if (valid())
      {
        storage_->acquire(index_);
      }
    }
    return *this;
  }

  ResourceHandle& operator=(ResourceHandle&& other) noexcept
  {
    if (this != &other)
    {
      release();
      index_ = other.index_;
      storage_ = other.storage_;
      other.storage_ = nullptr;
    }
    return *this;
  }

  T* operator->() { return &storage_->resources_[index_]; }
  T& operator*() { return storage_->resources_[index_]; }
  const T* operator->() const { return &*storage_->resources_[index_]; }
  const T& operator*() const { return *storage_->resources_[index_]; }

  [[nodiscard]] bool valid() const { return storage_ != nullptr; }
  explicit operator bool() const { return valid(); }

private:
  friend class ResourceStorage<T>;

  ResourceHandle(const uint32_t index, ResourceStorage<T>* storage) : index_(index), storage_(storage) {}

  void release()
  {
    if (valid())
    {
      storage_->release(index_);
    }
    storage_ = nullptr;
  }

  uint32_t index_ = 0;
  ResourceStorage<T>* storage_;
};

// concept for valid loaders. needs () operator that takes the right args and
// returns the right type
template<typename Loader, typename T, typename... Args>
concept LoaderFor = std::invocable<Loader, Args...> && std::same_as<std::invoke_result_t<Loader, Args...>, T>;

template<typename T>
class ResourceStorage
{
  friend class ResourceHandle<T>;
  using Handle = ResourceHandle<T>;

  std::vector<std::optional<T>> resources_;
  std::vector<uint32_t> ref_counts_;
  std::vector<std::string> keys_;
  std::vector<uint32_t> free_;

  std::unordered_map<std::string, uint32_t> key_to_index_;

  std::vector<ResourceCallback<T>> on_destroy_;

  void acquire(const uint32_t index) { ++ref_counts_.at(index); }

  void release(const uint32_t index)
  {
    assert(ref_counts_.at(index) > 0);
    if (--ref_counts_.at(index) == 0)
    {
      for (const auto& callback: on_destroy_)
      {
        callback(*resources_[index]);
      }
      key_to_index_.erase(keys_.at(index));
      keys_.at(index).clear();
      resources_[index].reset();
      free_.push_back(index);
    }
  }

public:
  template<typename Loader, typename... Args>
    requires LoaderFor<Loader, T, Args...>
  Handle load(const std::string& key, Loader&& loader, Args&&... args)
  {
    auto iter = key_to_index_.find(key);
    if (iter != key_to_index_.end())
    {
      ++ref_counts_.at(iter->second);
      return Handle{iter->second, this};
    }

    // load before touching indices because might throw exception
    T resource = std::forward<Loader>(loader)(std::forward<Args>(args)...);

    uint32_t index{};
    if (!free_.empty())
    {
      index = free_.back();
      free_.pop_back();
      resources_[index] = std::move(resource);
    } else
    {
      index = static_cast<uint32_t>(resources_.size());
      resources_.emplace_back(std::move(resource));
      ref_counts_.emplace_back();
      keys_.emplace_back();
    }

    ref_counts_.at(index) = 1;
    keys_.at(index) = key;
    key_to_index_[key] = index;

    return Handle{index, this};
  }

  Handle get(const std::string& key)
  {
    auto iter = key_to_index_.find(key);
    if (iter == key_to_index_.end())
    {
      return {};
    }
    ++ref_counts_.at(iter->second);
    return Handle{iter->second, this};
  }

  void AddOnDestroyCallback(const ResourceCallback<T>& callback) { on_destroy_.push_back(callback); }
};
