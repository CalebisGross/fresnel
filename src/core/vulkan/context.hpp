#pragma once

#include <kompute/Kompute.hpp>
#include <memory>
#include <string>
#include <vector>

namespace fresnel {

/**
 * VulkanContext - Manages Vulkan instance and Kompute manager
 *
 * This class wraps Kompute's Manager to provide a clean interface
 * for GPU compute operations. It handles device selection and
 * provides access to the underlying Kompute manager for creating
 * sequences and tensors.
 */
class VulkanContext {
public:
    struct DeviceInfo {
        std::string name;
        uint32_t device_id;
        uint64_t vram_bytes;
        bool is_discrete;
    };

    /**
     * Initialize Vulkan context with automatic device selection
     * Prefers discrete GPUs with most VRAM
     */
    VulkanContext();

    /**
     * Initialize with specific device index
     */
    explicit VulkanContext(uint32_t device_index);

    ~VulkanContext() = default;

    // Non-copyable, movable
    VulkanContext(const VulkanContext&) = delete;
    VulkanContext& operator=(const VulkanContext&) = delete;
    VulkanContext(VulkanContext&&) = default;
    VulkanContext& operator=(VulkanContext&&) = default;

    /**
     * Get the Kompute manager for creating tensors and sequences
     */
    kp::Manager& manager() { return *manager_; }
    const kp::Manager& manager() const { return *manager_; }

    /**
     * Get information about the current device
     */
    const DeviceInfo& device_info() const { return device_info_; }

    /**
     * List all available Vulkan devices
     */
    static std::vector<DeviceInfo> enumerate_devices();

    /**
     * Check if Vulkan is available on this system
     */
    static bool is_available();

private:
    std::unique_ptr<kp::Manager> manager_;
    DeviceInfo device_info_;

    void query_device_info();
};

} // namespace fresnel
