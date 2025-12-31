#include "core/pointcloud.hpp"
#include "core/depth/estimator.hpp"
#include "core/vulkan/context.hpp"
#include "core/renderer/renderer.hpp"
#include <iostream>
#include <fstream>
#include <cmath>

/**
 * Test suite for point cloud conversion
 *
 * Verifies:
 * 1. Point cloud creation from depth
 * 2. Conversion to Gaussians
 * 3. End-to-end image → depth → pointcloud → Gaussians → render
 */

bool test_pointcloud_creation() {
    std::cout << "[TEST] Point cloud creation... ";

    // Create a simple depth map
    fresnel::DepthMap depth(64, 64);
    for (uint32_t y = 0; y < 64; y++) {
        for (uint32_t x = 0; x < 64; x++) {
            // Depth increases from center (closer at center)
            float dx = x - 32.0f;
            float dy = y - 32.0f;
            float dist = std::sqrt(dx * dx + dy * dy) / 45.0f;
            depth.at(x, y) = dist;
        }
    }

    auto cloud = fresnel::PointCloud::from_depth(depth, nullptr);

    if (cloud.empty()) {
        std::cout << "FAIL (empty cloud)\n";
        return false;
    }

    std::cout << "PASS (" << cloud.size() << " points)\n";
    return true;
}

bool test_pointcloud_with_color() {
    std::cout << "[TEST] Point cloud with color... ";

    // Create depth and color
    fresnel::DepthMap depth(32, 32);
    fresnel::Image color(32, 32, 3);

    for (uint32_t y = 0; y < 32; y++) {
        for (uint32_t x = 0; x < 32; x++) {
            depth.at(x, y) = 0.5f;
            color.set_rgb(x, y, 1.0f, 0.0f, 0.0f); // Red
        }
    }

    auto cloud = fresnel::PointCloud::from_depth(depth, &color);

    if (cloud.empty()) {
        std::cout << "FAIL (empty cloud)\n";
        return false;
    }

    // Check color was applied
    if (cloud[0].color.r < 0.5f) {
        std::cout << "FAIL (color not applied)\n";
        return false;
    }

    std::cout << "PASS\n";
    return true;
}

bool test_pointcloud_to_gaussians() {
    std::cout << "[TEST] Point cloud to Gaussians... ";

    fresnel::PointCloud cloud;
    cloud.add(glm::vec3(0, 0, 0), glm::vec3(1, 0, 0));
    cloud.add(glm::vec3(1, 0, 0), glm::vec3(0, 1, 0));
    cloud.add(glm::vec3(0, 1, 0), glm::vec3(0, 0, 1));

    auto gaussians = cloud.to_gaussians(0.1f, 0.9f);

    if (gaussians.size() != 3) {
        std::cout << "FAIL (expected 3 gaussians, got " << gaussians.size() << ")\n";
        return false;
    }

    // Check first Gaussian
    if (gaussians[0].color.r < 0.5f) {
        std::cout << "FAIL (color not preserved)\n";
        return false;
    }

    std::cout << "PASS\n";
    return true;
}

bool test_pointcloud_normalize() {
    std::cout << "[TEST] Point cloud normalize... ";

    fresnel::PointCloud cloud;
    cloud.add(glm::vec3(-100, 0, 0), glm::vec3(1, 1, 1));
    cloud.add(glm::vec3(100, 0, 0), glm::vec3(1, 1, 1));
    cloud.add(glm::vec3(0, -50, 0), glm::vec3(1, 1, 1));
    cloud.add(glm::vec3(0, 50, 0), glm::vec3(1, 1, 1));

    cloud.normalize(4.0f);

    glm::vec3 min, max;
    cloud.get_bounds(min, max);

    float extent = std::max({max.x - min.x, max.y - min.y, max.z - min.z});

    if (std::abs(extent - 4.0f) > 0.1f) {
        std::cout << "FAIL (extent is " << extent << ", expected 4.0)\n";
        return false;
    }

    std::cout << "PASS (extent = " << extent << ")\n";
    return true;
}

bool test_end_to_end_pipeline() {
    std::cout << "[TEST] End-to-end pipeline... ";

    // Create synthetic image
    fresnel::Image image(128, 128, 3);
    for (uint32_t y = 0; y < 128; y++) {
        for (uint32_t x = 0; x < 128; x++) {
            // Create a colorful pattern
            float r = static_cast<float>(x) / 128.0f;
            float g = static_cast<float>(y) / 128.0f;
            float b = 0.5f;
            image.set_rgb(x, y, r, g, b);
        }
    }

    // Estimate depth
    auto estimator = fresnel::create_depth_estimator();
    auto depth = estimator->estimate(image);

    // Create point cloud
    auto cloud = fresnel::create_pointcloud_from_image(image, depth, 0.03f, 2);

    if (cloud.empty()) {
        std::cout << "FAIL (empty cloud from pipeline)\n";
        return false;
    }

    // Convert to Gaussians
    auto gaussians = cloud.to_gaussians();

    if (gaussians.empty()) {
        std::cout << "FAIL (empty gaussians)\n";
        return false;
    }

    std::cout << "PASS (" << cloud.size() << " points → " << gaussians.size() << " Gaussians)\n";
    return true;
}

bool test_render_pointcloud() {
    std::cout << "[TEST] Render point cloud... ";

    try {
        fresnel::VulkanContext ctx;
        fresnel::GaussianRenderer renderer(ctx);

        fresnel::RenderSettings settings;
        settings.width = 256;
        settings.height = 256;
        settings.background_color = glm::vec3(0.1f, 0.1f, 0.15f);
        renderer.init(settings);

        // Create a point cloud
        fresnel::PointCloud cloud;
        for (int i = 0; i < 500; i++) {
            float theta = i * 0.1f;
            float r = 1.0f + 0.3f * std::sin(theta * 3.0f);
            float x = r * std::cos(theta);
            float y = (i - 250) * 0.01f;
            float z = r * std::sin(theta);

            // Rainbow color
            float hue = static_cast<float>(i) / 500.0f;
            glm::vec3 color;
            if (hue < 0.33f) {
                color = glm::vec3(1.0f - hue * 3, hue * 3, 0);
            } else if (hue < 0.67f) {
                color = glm::vec3(0, 1.0f - (hue - 0.33f) * 3, (hue - 0.33f) * 3);
            } else {
                color = glm::vec3((hue - 0.67f) * 3, 0, 1.0f - (hue - 0.67f) * 3);
            }

            cloud.add(glm::vec3(x, y, z), color);
        }

        auto gaussians = cloud.to_gaussians(0.05f, 0.9f);
        renderer.upload_gaussians(gaussians);

        fresnel::Camera camera;
        camera.set_viewport(256, 256);
        camera.look_at(glm::vec3(0, 2, 5), glm::vec3(0, 0, 0));

        renderer.render(camera);

        // Save result
        auto rgba8 = renderer.get_image_rgba8();
        std::ofstream file("test_pointcloud_render.ppm", std::ios::binary);
        if (file) {
            file << "P6\n256 256\n255\n";
            for (size_t i = 0; i < rgba8.size(); i += 4) {
                file.put(static_cast<char>(rgba8[i]));
                file.put(static_cast<char>(rgba8[i + 1]));
                file.put(static_cast<char>(rgba8[i + 2]));
            }
            file.close();
        }

        std::cout << "PASS (saved test_pointcloud_render.ppm)\n";
        return true;

    } catch (const std::exception& e) {
        std::cout << "FAIL (" << e.what() << ")\n";
        return false;
    }
}

int main() {
    std::cout << "=== Fresnel Point Cloud Tests ===\n\n";

    if (!fresnel::VulkanContext::is_available()) {
        std::cerr << "WARNING: No Vulkan-capable GPU found, skipping render test\n";
    }

    int passed = 0;
    int failed = 0;

    auto run_test = [&](bool (*test)()) {
        if (test()) passed++;
        else failed++;
    };

    run_test(test_pointcloud_creation);
    run_test(test_pointcloud_with_color);
    run_test(test_pointcloud_to_gaussians);
    run_test(test_pointcloud_normalize);
    run_test(test_end_to_end_pipeline);
    run_test(test_render_pointcloud);

    std::cout << "\n=== Results: " << passed << " passed, " << failed << " failed ===\n";
    return failed > 0 ? 1 : 0;
}
