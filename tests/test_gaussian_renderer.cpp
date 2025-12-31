#include "core/vulkan/context.hpp"
#include "core/renderer/renderer.hpp"
#include "core/renderer/gaussian.hpp"
#include "core/renderer/camera.hpp"
#include <iostream>
#include <fstream>
#include <cmath>

/**
 * Test suite for Gaussian splatting renderer
 *
 * Verifies:
 * 1. Gaussian data structures
 * 2. Camera projection
 * 3. Renderer initialization
 * 4. Basic rendering
 * 5. Output image validity
 */

bool test_gaussian_struct() {
    std::cout << "[TEST] Gaussian3D struct... ";

    fresnel::Gaussian3D g;
    g.position = glm::vec3(1.0f, 2.0f, 3.0f);
    g.scale = glm::vec3(0.5f, 0.5f, 0.5f);
    g.rotation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f); // Identity
    g.color = glm::vec3(1.0f, 0.0f, 0.0f); // Red
    g.opacity = 0.9f;

    // Check covariance computation
    glm::mat3 cov = g.covariance();

    // For identity rotation and uniform scale 0.5:
    // Σ = R * S^2 * R^T = I * 0.25 * I = 0.25 * I
    bool cov_ok = true;
    float expected = 0.25f;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            float val = (i == j) ? expected : 0.0f;
            if (std::abs(cov[i][j] - val) > 1e-5f) {
                cov_ok = false;
            }
        }
    }

    if (!cov_ok) {
        std::cout << "FAIL (covariance mismatch)\n";
        return false;
    }

    // Note: Struct size doesn't need to be exact since we manually pack data for GPU upload
    // The renderer serializes Gaussians to 16 floats regardless of struct padding

    std::cout << "PASS\n";
    return true;
}

bool test_camera() {
    std::cout << "[TEST] Camera projection... ";

    fresnel::Camera cam;
    cam.set_viewport(800, 600);
    cam.look_at(glm::vec3(0, 0, 5), glm::vec3(0, 0, 0));

    // Project origin - should be at center
    glm::vec2 center = cam.project(glm::vec3(0, 0, 0));
    if (std::abs(center.x - 400.0f) > 1.0f || std::abs(center.y - 300.0f) > 1.0f) {
        std::cout << "FAIL (center at " << center.x << ", " << center.y << ")\n";
        return false;
    }

    // Project point to the right - should be right of center
    glm::vec2 right = cam.project(glm::vec3(1, 0, 0));
    if (right.x <= center.x) {
        std::cout << "FAIL (right point not right of center)\n";
        return false;
    }

    // Check depth
    float depth = cam.view_depth(glm::vec3(0, 0, 0));
    if (std::abs(depth - 5.0f) > 0.01f) {
        std::cout << "FAIL (depth is " << depth << ", expected 5)\n";
        return false;
    }

    std::cout << "PASS\n";
    return true;
}

bool test_gaussian_cloud() {
    std::cout << "[TEST] Gaussian cloud... ";

    auto cloud = fresnel::GaussianCloud::create_test_cloud(100, 2.0f);

    if (cloud.size() != 100) {
        std::cout << "FAIL (size is " << cloud.size() << ", expected 100)\n";
        return false;
    }

    // Check all Gaussians have valid data
    for (size_t i = 0; i < cloud.size(); i++) {
        const auto& g = cloud[i];
        if (std::isnan(g.position.x) || std::isnan(g.position.y) || std::isnan(g.position.z)) {
            std::cout << "FAIL (NaN position at " << i << ")\n";
            return false;
        }
        if (g.scale.x <= 0 || g.scale.y <= 0 || g.scale.z <= 0) {
            std::cout << "FAIL (invalid scale at " << i << ")\n";
            return false;
        }
        if (g.opacity < 0 || g.opacity > 1) {
            std::cout << "FAIL (invalid opacity at " << i << ")\n";
            return false;
        }
    }

    std::cout << "PASS\n";
    return true;
}

bool test_renderer_init() {
    std::cout << "[TEST] Renderer initialization... ";

    try {
        fresnel::VulkanContext ctx;
        fresnel::GaussianRenderer renderer(ctx);

        fresnel::RenderSettings settings;
        settings.width = 320;
        settings.height = 240;
        settings.background_color = glm::vec3(0.1f, 0.1f, 0.2f);

        renderer.init(settings);

        if (renderer.settings().width != 320 || renderer.settings().height != 240) {
            std::cout << "FAIL (settings not applied)\n";
            return false;
        }

        std::cout << "PASS\n";
        return true;
    } catch (const std::exception& e) {
        std::cout << "FAIL (" << e.what() << ")\n";
        return false;
    }
}

bool test_render_basic() {
    std::cout << "[TEST] Basic rendering... ";

    try {
        fresnel::VulkanContext ctx;
        fresnel::GaussianRenderer renderer(ctx);

        fresnel::RenderSettings settings;
        settings.width = 256;
        settings.height = 256;
        settings.background_color = glm::vec3(0.0f);

        renderer.init(settings);

        // Create a simple test cloud - single red Gaussian at origin
        fresnel::GaussianCloud cloud;
        cloud.add(fresnel::Gaussian3D(
            glm::vec3(0.0f, 0.0f, 0.0f),    // position
            glm::vec3(0.5f, 0.5f, 0.5f),    // scale
            glm::identity<glm::quat>(),     // rotation
            glm::vec3(1.0f, 0.0f, 0.0f),    // color (red)
            0.9f                             // opacity
        ));

        renderer.upload_gaussians(cloud);

        // Set up camera
        fresnel::Camera camera;
        camera.set_viewport(256, 256);
        camera.look_at(glm::vec3(0, 0, 3), glm::vec3(0, 0, 0));

        // Render
        const auto& pixels = renderer.render(camera);

        // Check we got pixels
        if (pixels.size() != 256 * 256 * 4) {
            std::cout << "FAIL (wrong pixel count: " << pixels.size() << ")\n";
            return false;
        }

        // Check center pixel is reddish (the Gaussian should be there)
        size_t center_idx = (128 * 256 + 128) * 4;
        float r = pixels[center_idx + 0];
        float g = pixels[center_idx + 1];
        float b = pixels[center_idx + 2];

        if (r < 0.1f) {
            std::cout << "FAIL (center pixel not red: R=" << r << " G=" << g << " B=" << b << ")\n";
            return false;
        }

        auto stats = renderer.stats();
        std::cout << "PASS (render: " << stats.total_time_ms << " ms)\n";
        return true;
    } catch (const std::exception& e) {
        std::cout << "FAIL (" << e.what() << ")\n";
        return false;
    }
}

bool test_render_many() {
    std::cout << "[TEST] Rendering 1000 Gaussians... ";

    try {
        fresnel::VulkanContext ctx;
        fresnel::GaussianRenderer renderer(ctx);

        fresnel::RenderSettings settings;
        settings.width = 512;
        settings.height = 512;
        settings.background_color = glm::vec3(0.05f, 0.05f, 0.1f);

        renderer.init(settings);

        // Create test cloud
        auto cloud = fresnel::GaussianCloud::create_test_cloud(1000, 3.0f);
        renderer.upload_gaussians(cloud);

        // Set up camera
        fresnel::Camera camera;
        camera.set_viewport(512, 512);
        camera.look_at(glm::vec3(0, 0, 8), glm::vec3(0, 0, 0));

        // Render
        const auto& pixels = renderer.render(camera);

        // Check we got valid pixels
        bool has_content = false;
        for (size_t i = 0; i < pixels.size(); i += 4) {
            if (pixels[i] > 0.1f || pixels[i+1] > 0.1f || pixels[i+2] > 0.1f) {
                has_content = true;
                break;
            }
        }

        if (!has_content) {
            std::cout << "FAIL (no visible content)\n";
            return false;
        }

        auto stats = renderer.stats();
        std::cout << "PASS (" << stats.total_gaussians << " Gaussians, "
                  << stats.total_time_ms << " ms total, "
                  << stats.render_time_ms << " ms render)\n";
        return true;
    } catch (const std::exception& e) {
        std::cout << "FAIL (" << e.what() << ")\n";
        return false;
    }
}

bool test_save_image() {
    std::cout << "[TEST] Save rendered image... ";

    try {
        fresnel::VulkanContext ctx;
        fresnel::GaussianRenderer renderer(ctx);

        fresnel::RenderSettings settings;
        settings.width = 512;
        settings.height = 512;
        settings.background_color = glm::vec3(0.1f, 0.1f, 0.15f);

        renderer.init(settings);

        // Create colorful test cloud
        auto cloud = fresnel::GaussianCloud::create_test_cloud(500, 2.5f);
        renderer.upload_gaussians(cloud);

        fresnel::Camera camera;
        camera.set_viewport(512, 512);
        camera.look_at(glm::vec3(0, 2, 6), glm::vec3(0, 0, 0));

        renderer.render(camera);

        // Save as PPM (simple format, no external deps)
        auto rgba8 = renderer.get_image_rgba8();

        std::ofstream file("test_render.ppm", std::ios::binary);
        if (!file) {
            std::cout << "FAIL (couldn't create file)\n";
            return false;
        }

        file << "P6\n" << settings.width << " " << settings.height << "\n255\n";
        for (size_t i = 0; i < rgba8.size(); i += 4) {
            file.put(rgba8[i]);     // R
            file.put(rgba8[i+1]);   // G
            file.put(rgba8[i+2]);   // B
        }
        file.close();

        std::cout << "PASS (saved test_render.ppm)\n";
        return true;
    } catch (const std::exception& e) {
        std::cout << "FAIL (" << e.what() << ")\n";
        return false;
    }
}

int main() {
    std::cout << "=== Fresnel Gaussian Renderer Tests ===\n\n";

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

    run_test(test_gaussian_struct);
    run_test(test_camera);
    run_test(test_gaussian_cloud);
    run_test(test_renderer_init);
    run_test(test_render_basic);
    run_test(test_render_many);
    run_test(test_save_image);

    std::cout << "\n=== Results: " << passed << " passed, " << failed << " failed ===\n";
    return failed > 0 ? 1 : 0;
}
