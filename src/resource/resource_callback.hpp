#pragma once

template<typename R>
class ResourceCallback {
public:
  template<auto MemberFnc, typename T>
  static ResourceCallback create(T* obj)
  {
    return {[](void* ctx, const R& resource) { (static_cast<T*>(ctx)->*MemberFnc)(resource); }, obj};
  }

  void operator()(const R& resource) const { fnc_(ctx_, resource); }

private:
  ResourceCallback(void (*fnc)(void*, const R&), void* ctx) : fnc_(fnc), ctx_(ctx) {}

  void (*fnc_)(void* ctx, const R& resource);
  void* ctx_;
};
