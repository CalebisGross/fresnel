#pragma once

#include "core/vulkan/context.hpp"
#include "core/renderer/renderer.hpp"
#include "core/renderer/camera.hpp"
#include "core/renderer/gaussian.hpp"

#include <GLFW/glfw3.h>
#include <imgui.h>
#include <glm/glm.hpp>

#include <memory>
#include <string>

namespace fresnel {

/**
 * Viewer - Interactive ImGui-based 3D viewer for Gaussian splatting
 *
 * Features:
 * - Real-time Gaussian rendering via Vulkan compute
 * - Orbit camera with mouse controls
 * - ImGui UI panels for settings and stats
 * - OpenGL texture display of rendered frames
 */
class Viewer {
public:
    struct Settings {
        int window_width = 1280;
        int window_height = 720;
        std::string title = "Fresnel Viewer";
        glm::vec3 background_color = glm::vec3(0.1f, 0.1f, 0.15f);
        int render_width = 800;
        int render_height = 600;
    };

    Viewer();
    ~Viewer();

    // Non-copyable
    Viewer(const Viewer&) = delete;
    Viewer& operator=(const Viewer&) = delete;

    /**
     * Initialize the viewer window and renderer
     */
    bool init(const Settings& settings);

    /**
     * Load Gaussians to display
     */
    void load_gaussians(const GaussianCloud& cloud);

    /**
     * Create a test cloud for demonstration
     */
    void load_test_cloud(size_t count = 1000, float extent = 3.0f);

    /**
     * Run the main loop until window is closed
     */
    void run();

    /**
     * Check if viewer is still running
     */
    bool is_running() const;

    /**
     * Request viewer to close
     */
    void close();

private:
    // Window and context
    GLFWwindow* window_ = nullptr;
    std::unique_ptr<VulkanContext> vk_context_;
    std::unique_ptr<GaussianRenderer> renderer_;

    // Camera state
    Camera camera_;
    glm::vec3 camera_target_ = glm::vec3(0.0f);
    float camera_distance_ = 5.0f;
    float camera_yaw_ = 0.0f;
    float camera_pitch_ = 0.3f;

    // Mouse interaction state
    bool mouse_dragging_ = false;
    double last_mouse_x_ = 0.0;
    double last_mouse_y_ = 0.0;

    // OpenGL texture for display
    unsigned int render_texture_ = 0;

    // Settings and state
    Settings settings_;
    bool running_ = false;
    bool needs_rerender_ = true;

    // Stats for display
    struct Stats {
        float fps = 0.0f;
        float frame_time_ms = 0.0f;
        float render_time_ms = 0.0f;
        int gaussian_count = 0;
    } stats_;

    // Internal methods
    void init_imgui();
    void shutdown_imgui();
    void init_render_texture();
    void update_camera();
    void process_input();
    void render_frame();
    void render_ui();

    // Callbacks
    static void glfw_error_callback(int error, const char* description);
    static void glfw_scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
};

} // namespace fresnel
