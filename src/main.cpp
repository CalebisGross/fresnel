#include "core/vulkan/context.hpp"
#include "core/compute/pipeline.hpp"
#include <iostream>
#include <iomanip>

void print_device_info(const fresnel::VulkanContext::DeviceInfo& info) {
    std::cout << "  Name: " << info.name << "\n";
    std::cout << "  VRAM: " << std::fixed << std::setprecision(2)
              << (info.vram_bytes / (1024.0 * 1024.0 * 1024.0)) << " GB\n";
    std::cout << "  Type: " << (info.is_discrete ? "Discrete" : "Integrated") << "\n";
}

int main(int argc, char* argv[]) {
    std::cout << "=== Fresnel: Single-Image to 3D Reconstruction ===\n\n";

    // Check Vulkan availability
    if (!fresnel::VulkanContext::is_available()) {
        std::cerr << "ERROR: No Vulkan-capable GPU found!\n";
        return 1;
    }

    // List available devices
    std::cout << "Available Vulkan devices:\n";
    auto devices = fresnel::VulkanContext::enumerate_devices();
    for (size_t i = 0; i < devices.size(); i++) {
        std::cout << "\n[" << i << "] ";
        print_device_info(devices[i]);
    }

    // Initialize context (auto-selects best device)
    std::cout << "\nInitializing Vulkan context...\n";
    fresnel::VulkanContext ctx;

    std::cout << "\nUsing device:\n";
    print_device_info(ctx.device_info());

    // Quick compute test
    std::cout << "\nRunning compute shader test...\n";
    fresnel::ComputePipeline pipeline(ctx);

    // Simple shader that doubles each element
    const std::string shader_src = R"(
        #version 450
        layout(local_size_x = 256) in;

        layout(binding = 0) buffer InputBuffer {
            float data[];
        } input_buf;

        layout(binding = 1) buffer OutputBuffer {
            float data[];
        } output_buf;

        void main() {
            uint idx = gl_GlobalInvocationID.x;
            output_buf.data[idx] = input_buf.data[idx] * 2.0;
        }
    )";

    try {
        auto spirv = fresnel::compile_glsl(shader_src);
        std::cout << "  Shader compiled: " << spirv.size() << " words of SPIR-V\n";

        // Create test data
        std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
        auto input_tensor = pipeline.create_tensor<float>(std::span(input_data));
        auto output_tensor = pipeline.create_tensor<float>(input_data.size());

        // Upload input
        pipeline.sync_to_device({input_tensor});

        // Run shader
        pipeline.dispatch(spirv, {input_tensor, output_tensor}, (input_data.size() + 255) / 256);

        // Download output
        pipeline.sync_to_host({output_tensor});

        // Verify
        auto result = output_tensor->vector();
        std::cout << "  Input:  [";
        for (size_t i = 0; i < input_data.size(); i++) {
            std::cout << input_data[i] << (i < input_data.size()-1 ? ", " : "");
        }
        std::cout << "]\n";

        std::cout << "  Output: [";
        for (size_t i = 0; i < result.size(); i++) {
            std::cout << result[i] << (i < result.size()-1 ? ", " : "");
        }
        std::cout << "]\n";

        bool correct = true;
        for (size_t i = 0; i < input_data.size(); i++) {
            if (result[i] != input_data[i] * 2.0f) {
                correct = false;
                break;
            }
        }

        std::cout << "  Result: " << (correct ? "PASS" : "FAIL") << "\n";

    } catch (const std::exception& e) {
        std::cerr << "  ERROR: " << e.what() << "\n";
        return 1;
    }

    std::cout << "\nVulkan compute pipeline ready!\n";
    return 0;
}
