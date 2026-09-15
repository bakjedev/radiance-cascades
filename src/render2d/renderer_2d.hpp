#pragma once
#include "backend/device.hpp"
#include "backend/frame.hpp"
#include "backend/instance.hpp"
#include "backend/swapchain.hpp"
#include "framework/context.hpp"

class Window;
class EventDispatcher;

constexpr uint32_t frames_in_flight = 2;

struct FwrkAllocator : fwrk::Allocator {
    std::optional<fwrk::PhysicalImage> create_image( const fwrk::ImageCreateInfo& img_info ) override;
    std::optional<fwrk::PhysicalBuffer> create_buffer( const fwrk::BufferCreateInfo& buf_info ) override;
    void destroy_image( fwrk::PhysicalImage& img ) override;
    void destroy_buffer( fwrk::PhysicalBuffer& buf ) override;

    fwrk::flat_hash_map<VkImage, VmaAllocation> image_to_allocation;
    fwrk::flat_hash_map<VkBuffer, VmaAllocation> buffer_to_allocation;
    VmaAllocator allocator;

    explicit FwrkAllocator( VmaAllocator alc ) : allocator(alc) {
    }
};

class Renderer2D {
public:
    Renderer2D( Window& window, EventDispatcher& event_dispatcher );
    ~Renderer2D();

    void render();

private:
    bool begin_frame();
    void run_frame();
    void end_frame();

    Window& window_;
    EventDispatcher& event_dispatcher_;

    Instance instance_;
    Device device_;
    Swapchain swapchain_;
    std::array<Frame, frames_in_flight> frames_;
    std::vector<vk::UniqueSemaphore> submit_semaphores_;

    uint32_t current_frame_{};
    uint32_t image_index_{};

    FwrkAllocator fwrk_allocator_;
    fwrk::Context context_;

    static VkSurfaceKHR create_surface( const Window& window, const Instance& instance );
};
