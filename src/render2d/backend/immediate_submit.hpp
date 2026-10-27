#pragma once
#include "device.hpp"

class ImmediateSubmit {
public:
  explicit ImmediateSubmit(const Device& device) : device_(&device)
  {
    pool_ =
        device.get().createCommandPoolUnique({vk::CommandPoolCreateFlagBits::eTransient, device.get_queue_family()});
    cmd_ = device.get().allocateCommandBuffers({pool_.get(), vk::CommandBufferLevel::ePrimary, 1}).front();
    fence_ = device.get().createFenceUnique({});
  }

  template<std::invocable<vk::CommandBuffer> F>
  void submit(F&& func)
  {
    device_->get().resetCommandPool(pool_.get());
    cmd_.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    func(cmd_);
    cmd_.end();

    vk::CommandBufferSubmitInfo cmdInfo{cmd_};
    vk::SubmitInfo2 info{};
    info.setCommandBufferInfos(cmdInfo);
    device_->get_queue().submit2(info, fence_.get());

    const auto result = device_->get().waitForFences(fence_.get(), vk::True, UINT64_MAX);
    if (result != vk::Result::eSuccess) {
      throw std::runtime_error("Failed to wait for fences for immediate submit");
    }

    device_->get().resetFences(fence_.get());
  }

private:
  const Device* device_;
  vk::UniqueCommandPool pool_;
  vk::CommandBuffer cmd_;
  vk::UniqueFence fence_;
};
