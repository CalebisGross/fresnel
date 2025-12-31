#include "core/vulkan/context.hpp"
#include "core/compute/pipeline.hpp"
#include <iostream>
#include <cassert>
#include <cmath>

/**
 * Test suite for Vulkan compute pipeline
 *
 * Verifies:
 * 1. Device enumeration
 * 2. Context initialization
 * 3. Tensor creation and data transfer
 * 4. Compute shader execution
 * 5. Multiple dispatch patterns
 */

bool test_device_enumeration() {
    std::cout << "[TEST] Device enumeration... ";

    auto devices = fresnel::VulkanContext::enumerate_devices();
    if (devices.empty()) {
        std::cout << "SKIP (no devices)\n";
        return true;
    }

    for (const auto& dev : devices) {
        if (dev.name.empty()) {
            std::cout << "FAIL (empty device name)\n";
            return false;
        }
    }

    std::cout << "PASS (" << devices.size() << " device(s))\n";
    return true;
}

bool test_context_init() {
    std::cout << "[TEST] Context initialization... ";

    try {
        fresnel::VulkanContext ctx;
        if (ctx.device_info().name.empty()) {
            std::cout << "FAIL (no device name)\n";
            return false;
        }
        std::cout << "PASS (" << ctx.device_info().name << ")\n";
        return true;
    } catch (const std::exception& e) {
        std::cout << "FAIL (" << e.what() << ")\n";
        return false;
    }
}

bool test_tensor_operations() {
    std::cout << "[TEST] Tensor operations... ";

    try {
        fresnel::VulkanContext ctx;
        fresnel::ComputePipeline pipeline(ctx);

        // Create tensor with data
        std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
        auto tensor = pipeline.create_tensor<float>(std::span(data));

        // Sync to device and back
        pipeline.sync_to_device({tensor});
        pipeline.sync_to_host({tensor});

        // Verify data integrity
        auto result = tensor->vector();
        for (size_t i = 0; i < data.size(); i++) {
            if (result[i] != data[i]) {
                std::cout << "FAIL (data mismatch at " << i << ")\n";
                return false;
            }
        }

        std::cout << "PASS\n";
        return true;
    } catch (const std::exception& e) {
        std::cout << "FAIL (" << e.what() << ")\n";
        return false;
    }
}

bool test_simple_compute() {
    std::cout << "[TEST] Simple compute shader... ";

    const std::string shader_src = R"(
        #version 450
        layout(local_size_x = 64) in;

        layout(binding = 0) buffer InputBuffer {
            float data[];
        } input_buf;

        layout(binding = 1) buffer OutputBuffer {
            float data[];
        } output_buf;

        void main() {
            uint idx = gl_GlobalInvocationID.x;
            output_buf.data[idx] = input_buf.data[idx] * 2.0 + 1.0;
        }
    )";

    try {
        fresnel::VulkanContext ctx;
        fresnel::ComputePipeline pipeline(ctx);

        auto spirv = fresnel::compile_glsl(shader_src);

        std::vector<float> input = {0.0f, 1.0f, 2.0f, 3.0f};
        auto in_tensor = pipeline.create_tensor<float>(std::span(input));
        auto out_tensor = pipeline.create_tensor<float>(input.size());

        pipeline.sync_to_device({in_tensor});
        pipeline.dispatch(spirv, {in_tensor, out_tensor}, 1);
        pipeline.sync_to_host({out_tensor});

        auto result = out_tensor->vector();
        for (size_t i = 0; i < input.size(); i++) {
            float expected = input[i] * 2.0f + 1.0f;
            if (std::abs(result[i] - expected) > 1e-6f) {
                std::cout << "FAIL (expected " << expected << ", got " << result[i] << ")\n";
                return false;
            }
        }

        std::cout << "PASS\n";
        return true;
    } catch (const std::exception& e) {
        std::cout << "FAIL (" << e.what() << ")\n";
        return false;
    }
}

bool test_large_dispatch() {
    std::cout << "[TEST] Large dispatch (1M elements)... ";

    const std::string shader_src = R"(
        #version 450
        layout(local_size_x = 256) in;

        layout(binding = 0) buffer Buffer {
            float data[];
        } buf;

        void main() {
            uint idx = gl_GlobalInvocationID.x;
            buf.data[idx] = sqrt(float(idx));
        }
    )";

    try {
        fresnel::VulkanContext ctx;
        fresnel::ComputePipeline pipeline(ctx);

        auto spirv = fresnel::compile_glsl(shader_src);

        const size_t N = 1000000;
        auto tensor = pipeline.create_tensor<float>(N);

        pipeline.dispatch(spirv, {tensor}, (N + 255) / 256);
        pipeline.sync_to_host({tensor});

        // Spot check a few values
        auto result = tensor->vector();
        bool ok = true;
        for (size_t i : {0, 100, 10000, 999999}) {
            float expected = std::sqrt(static_cast<float>(i));
            if (std::abs(result[i] - expected) > 1e-3f) {
                std::cout << "FAIL (at " << i << ": expected " << expected << ", got " << result[i] << ")\n";
                ok = false;
                break;
            }
        }

        if (ok) std::cout << "PASS\n";
        return ok;
    } catch (const std::exception& e) {
        std::cout << "FAIL (" << e.what() << ")\n";
        return false;
    }
}

int main() {
    std::cout << "=== Fresnel Vulkan Compute Tests ===\n\n";

    if (!fresnel::VulkanContext::is_available()) {
        std::cerr << "ERROR: No Vulkan-capable GPU found!\n";
        return 1;
    }

    int passed = 0;
    int failed = 0;

    auto run_test = [&](bool (*test)()) {
        if (test()) passed++;
        else failed++;
    };

    run_test(test_device_enumeration);
    run_test(test_context_init);
    run_test(test_tensor_operations);
    run_test(test_simple_compute);
    run_test(test_large_dispatch);

    std::cout << "\n=== Results: " << passed << " passed, " << failed << " failed ===\n";
    return failed > 0 ? 1 : 0;
}
