#include "context.hpp"
#include <stdexcept>
#include <algorithm>
#include <iostream>

namespace fresnel {

VulkanContext::VulkanContext() {
    // Let Kompute select the best device (usually first discrete GPU)
    manager_ = std::make_unique<kp::Manager>();
    query_device_info();
}

VulkanContext::VulkanContext(uint32_t device_index) {
    manager_ = std::make_unique<kp::Manager>(device_index);
    query_device_info();
}

void VulkanContext::query_device_info() {
    // Get device properties directly from Kompute
    vk::PhysicalDeviceProperties props = manager_->getDeviceProperties();

    device_info_.name = std::string(props.deviceName.data());
    device_info_.device_id = props.deviceID;
    device_info_.is_discrete = (props.deviceType == vk::PhysicalDeviceType::eDiscreteGpu);

    // For VRAM, we query from the devices list
    auto devices = manager_->listDevices();
    if (!devices.empty()) {
        // Use first device's memory properties (same as what Kompute selected)
        vk::PhysicalDeviceMemoryProperties mem_props = devices[0].getMemoryProperties();

        device_info_.vram_bytes = 0;
        for (uint32_t i = 0; i < mem_props.memoryHeapCount; i++) {
            if (mem_props.memoryHeaps[i].flags & vk::MemoryHeapFlagBits::eDeviceLocal) {
                device_info_.vram_bytes = std::max(device_info_.vram_bytes,
                                                   static_cast<uint64_t>(mem_props.memoryHeaps[i].size));
            }
        }
    }
}

std::vector<VulkanContext::DeviceInfo> VulkanContext::enumerate_devices() {
    std::vector<DeviceInfo> result;

    try {
        // Create a temporary manager just to list devices
        kp::Manager temp_mgr;
        auto vk_devices = temp_mgr.listDevices();

        for (const auto& vk_dev : vk_devices) {
            vk::PhysicalDeviceProperties props = vk_dev.getProperties();
            vk::PhysicalDeviceMemoryProperties mem_props = vk_dev.getMemoryProperties();

            DeviceInfo info;
            info.name = std::string(props.deviceName.data());
            info.device_id = props.deviceID;
            info.is_discrete = (props.deviceType == vk::PhysicalDeviceType::eDiscreteGpu);

            info.vram_bytes = 0;
            for (uint32_t j = 0; j < mem_props.memoryHeapCount; j++) {
                if (mem_props.memoryHeaps[j].flags & vk::MemoryHeapFlagBits::eDeviceLocal) {
                    info.vram_bytes = std::max(info.vram_bytes,
                                               static_cast<uint64_t>(mem_props.memoryHeaps[j].size));
                }
            }

            result.push_back(info);
        }
    } catch (...) {
        // If enumeration fails, return empty list
    }

    return result;
}

bool VulkanContext::is_available() {
    try {
        kp::Manager temp_mgr;
        return !temp_mgr.listDevices().empty();
    } catch (...) {
        return false;
    }
}

} // namespace fresnel
