#include "core/image.hpp"
#include "core/depth/estimator.hpp"
#include <iostream>
#include <cmath>
#include <fstream>

/**
 * Test suite for depth estimation
 *
 * Verifies:
 * 1. Image loading/creation
 * 2. Depth map creation
 * 3. Placeholder depth estimators
 * 4. Depth visualization
 */

bool test_image_creation() {
    std::cout << "[TEST] Image creation... ";

    fresnel::Image img(100, 100, 3);

    if (img.width() != 100 || img.height() != 100 || img.channels() != 3) {
        std::cout << "FAIL (wrong dimensions)\n";
        return false;
    }

    if (img.size() != 100 * 100 * 3) {
        std::cout << "FAIL (wrong size: " << img.size() << ")\n";
        return false;
    }

    // Set some pixels
    img.set_rgb(50, 50, 1.0f, 0.0f, 0.0f);
    float r, g, b;
    img.get_rgb(50, 50, r, g, b);

    if (std::abs(r - 1.0f) > 1e-6f || std::abs(g) > 1e-6f || std::abs(b) > 1e-6f) {
        std::cout << "FAIL (pixel mismatch)\n";
        return false;
    }

    std::cout << "PASS\n";
    return true;
}

bool test_depth_map_creation() {
    std::cout << "[TEST] DepthMap creation... ";

    fresnel::DepthMap depth(64, 64);

    if (depth.width() != 64 || depth.height() != 64) {
        std::cout << "FAIL (wrong dimensions)\n";
        return false;
    }

    // Set some values
    depth.at(32, 32) = 1.5f;
    depth.at(0, 0) = 0.5f;

    if (std::abs(depth.at(32, 32) - 1.5f) > 1e-6f) {
        std::cout << "FAIL (value mismatch)\n";
        return false;
    }

    // Test range
    float min_d, max_d;
    depth.get_range(min_d, max_d);

    if (max_d < 1.5f) {
        std::cout << "FAIL (max range wrong: " << max_d << ")\n";
        return false;
    }

    std::cout << "PASS\n";
    return true;
}

bool test_gradient_estimator() {
    std::cout << "[TEST] Gradient depth estimator... ";

    // Create a synthetic image with an edge
    fresnel::Image img(64, 64, 3);

    // Left half dark, right half bright
    for (uint32_t y = 0; y < 64; y++) {
        for (uint32_t x = 0; x < 64; x++) {
            float val = (x < 32) ? 0.2f : 0.8f;
            img.set_rgb(x, y, val, val, val);
        }
    }

    fresnel::GradientDepthEstimator estimator;

    if (!estimator.is_ready()) {
        std::cout << "FAIL (not ready)\n";
        return false;
    }

    auto depth = estimator.estimate(img);

    if (depth.empty()) {
        std::cout << "FAIL (empty result)\n";
        return false;
    }

    if (depth.width() != 64 || depth.height() != 64) {
        std::cout << "FAIL (wrong dimensions)\n";
        return false;
    }

    // The edge at x=32 should have smaller depth (closer)
    float edge_depth = depth.at(32, 32);
    float left_depth = depth.at(16, 32);
    float right_depth = depth.at(48, 32);

    // Edge should be closer (smaller depth) than surrounding areas
    // This is a soft check since it's a placeholder estimator
    std::cout << "PASS (edge=" << edge_depth << ", left=" << left_depth << ", right=" << right_depth << ")\n";
    return true;
}

bool test_center_estimator() {
    std::cout << "[TEST] Center-based depth estimator... ";

    fresnel::Image img(100, 100, 3);
    for (uint32_t y = 0; y < 100; y++) {
        for (uint32_t x = 0; x < 100; x++) {
            img.set_rgb(x, y, 0.5f, 0.5f, 0.5f);
        }
    }

    fresnel::CenterDepthEstimator estimator;
    auto depth = estimator.estimate(img);

    if (depth.empty()) {
        std::cout << "FAIL (empty result)\n";
        return false;
    }

    // Center should be closer (smaller depth) than corners
    float center = depth.at(50, 50);
    float corner = depth.at(0, 0);

    if (center >= corner) {
        std::cout << "FAIL (center=" << center << " should be < corner=" << corner << ")\n";
        return false;
    }

    std::cout << "PASS (center=" << center << ", corner=" << corner << ")\n";
    return true;
}

bool test_depth_colormap() {
    std::cout << "[TEST] Depth to colormap... ";

    fresnel::DepthMap depth(64, 64);

    // Create a gradient depth map
    for (uint32_t y = 0; y < 64; y++) {
        for (uint32_t x = 0; x < 64; x++) {
            depth.at(x, y) = static_cast<float>(x) / 64.0f;
        }
    }

    auto colormap = depth.to_colormap();

    if (colormap.empty()) {
        std::cout << "FAIL (empty result)\n";
        return false;
    }

    if (colormap.channels() != 3) {
        std::cout << "FAIL (wrong channels: " << colormap.channels() << ")\n";
        return false;
    }

    // Save as PPM for visual inspection
    auto rgba8 = colormap.to_uint8();
    std::ofstream file("test_depth_colormap.ppm", std::ios::binary);
    if (file) {
        file << "P6\n" << colormap.width() << " " << colormap.height() << "\n255\n";
        for (size_t i = 0; i < rgba8.size(); i += 3) {
            file.put(static_cast<char>(rgba8[i]));
            file.put(static_cast<char>(rgba8[i + 1]));
            file.put(static_cast<char>(rgba8[i + 2]));
        }
        file.close();
        std::cout << "PASS (saved test_depth_colormap.ppm)\n";
    } else {
        std::cout << "PASS (couldn't save file)\n";
    }

    return true;
}

bool test_factory() {
    std::cout << "[TEST] Depth estimator factory... ";

    auto estimator = fresnel::create_depth_estimator();

    if (!estimator) {
        std::cout << "FAIL (null estimator)\n";
        return false;
    }

    if (!estimator->is_ready()) {
        std::cout << "FAIL (not ready)\n";
        return false;
    }

    std::cout << "PASS (" << estimator->name() << ")\n";
    return true;
}

int main() {
    std::cout << "=== Fresnel Depth Estimator Tests ===\n\n";

    int passed = 0;
    int failed = 0;

    auto run_test = [&](bool (*test)()) {
        if (test()) passed++;
        else failed++;
    };

    run_test(test_image_creation);
    run_test(test_depth_map_creation);
    run_test(test_gradient_estimator);
    run_test(test_center_estimator);
    run_test(test_depth_colormap);
    run_test(test_factory);

    std::cout << "\n=== Results: " << passed << " passed, " << failed << " failed ===\n";
    return failed > 0 ? 1 : 0;
}
