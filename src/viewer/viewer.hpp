#pragma once

#include "core/vulkan/context.hpp"
#include "core/renderer/renderer.hpp"
#include "core/renderer/camera.hpp"
#include "core/renderer/gaussian.hpp"
#include "core/image.hpp"
#include "core/depth/estimator.hpp"
#include "core/features/feature_extractor.hpp"
#include "core/decoder/gaussian_decoder.hpp"
#include "core/pointcloud.hpp"

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
     * Load an image file and convert to 3D point cloud
     * Uses Depth Anything V2 for depth estimation
     */
    bool load_image(const std::string& path);

    /**
     * Reprocess the currently loaded image with current quality settings
     * @param preview If true, use faster preview settings
     */
    void reprocess_image(bool preview = false);

    /**
     * Export current view as high-quality image
     */
    bool export_image(const std::string& path);

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
        std::string depth_estimator_name;
        std::string loaded_image;
    } stats_;

    // Quality settings for image loading
    struct QualitySettings {
        int subsample = 1;           // 1 = full resolution, 2 = half, etc.
        float gaussian_size = 0.008f; // Size of each Gaussian splat
        float opacity = 0.9f;         // Gaussian opacity
        float depth_scale = 2.5f;     // How much depth "pops" out
        float depth_exponent = 0.7f;  // Depth curve (< 1 = more foreground detail)
        int max_gaussians = 500000;   // Cap to prevent slowdown
        bool auto_quality = true;     // Auto-adjust based on image size

        // SAAG (Surface-Aligned Anisotropic Gaussians) settings
        bool use_saag = true;         // Enable surface-aligned Gaussians
        float aspect_ratio = 5.0f;    // Gaussian flatness (higher = flatter discs)
        float edge_threshold = 0.15f; // Edge detection sensitivity
        float edge_shrink = 0.3f;     // Shrink factor at edges
        float gradient_scale = 50.0f;  // Normal computation sensitivity (was 2.0 - too small for [0,1] depth!)
        float normal_strength = 1.0f; // Alignment strength (0=spheres, 1=full)

        // Silhouette Wrapping settings (solves 2.5D problem)
        bool silhouette_wrap = true;   // Enable silhouette wrapping
        int wrap_layers = 3;           // Number of wrap layers (more = better side coverage)
        float wrap_spacing = 0.5f;     // Spacing between layers (relative to gaussian_size)
        float wrap_opacity = 0.7f;     // Opacity decay per layer
        float wrap_edge_threshold = 0.15f; // Gradient threshold for silhouette detection

        // Volumetric Shell settings (true 3D from any angle)
        bool volumetric_shell = true;    // Enable volumetric shell
        float shell_thickness = 0.3f;    // Shell thickness (scene-relative)
        float back_opacity = 0.6f;       // Back surface opacity
        float back_darken = 0.8f;        // Darken back surface (simulates less light)
        bool connect_walls = true;       // Add walls at silhouettes
        int wall_segments = 3;           // Segments between front and back
        float wall_opacity = 0.5f;       // Wall opacity
        float shell_edge_threshold = 0.1f; // Only shell where gradient > threshold (silhouettes)

        // Adaptive Density settings (fills gaps at edges to break stripe pattern)
        bool adaptive_density = true;     // Enable density adaptation
        float density_threshold = 0.08f;  // Gradient threshold for extra Gaussians
        int density_extra = 4;            // Extra Gaussians per edge point
        float density_jitter = 0.6f;      // Position randomness

        // Learned Decoder settings (replaces SAAG with ML model)
        bool use_learned_decoder = true; // Use learned Gaussian decoder when available
    } quality_;

    // Performance state
    bool pending_reprocess_ = false;  // Deferred reprocessing
    int preview_subsample_ = 4;       // Fast preview during interaction
    bool is_interacting_ = false;     // True during slider drag
    GaussianCloud full_quality_cloud_;  // Cached full-quality version

    // Depth estimator
    std::unique_ptr<DepthEstimator> depth_estimator_;

    // Feature extractor for learned decoder
    std::unique_ptr<FeatureExtractor> feature_extractor_;

    // Learned Gaussian decoder
    std::unique_ptr<GaussianDecoder> gaussian_decoder_;

    // Stored image/depth for re-processing with different settings
    Image loaded_image_;
    DepthMap loaded_depth_;

    // Stored features for learned decoder
    FeatureMap loaded_features_;

    // Image path input buffer for UI
    char image_path_buffer_[512] = "";
    char export_path_buffer_[512] = "output.ppm";

    // Internal helper for smart subsample calculation
    int calculate_auto_subsample() const;

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
