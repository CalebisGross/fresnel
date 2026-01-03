#pragma once

#include "gaussian.hpp"
#include "camera.hpp"
#include "../vulkan/context.hpp"
#include "../compute/pipeline.hpp"
#include "../compute/radix_sort.hpp"
#include <memory>
#include <vector>

namespace fresnel {

/**
 * RenderSettings - Configuration for the Gaussian renderer
 */
struct RenderSettings {
    uint32_t width = 800;
    uint32_t height = 600;
    uint32_t tile_size = 16;        // Tile size for tile-based rendering
    float gaussian_scale = 1.0f;    // Global scale multiplier for Gaussians
    float opacity_threshold = 1.0f / 255.0f;  // Minimum opacity to render
    bool sort_enabled = true;       // Enable depth sorting
    glm::vec3 background_color = glm::vec3(0.0f);
};

/**
 * GaussianRenderer - Tile-based Gaussian splatting renderer
 *
 * Implements the core Gaussian splatting algorithm:
 * 1. Project 3D Gaussians to 2D screen space
 * 2. Compute 2D covariance from 3D covariance via Jacobian
 * 3. Sort Gaussians by depth (front-to-back for alpha compositing)
 * 4. Tile-based rasterization for efficiency
 * 5. Alpha blending with proper compositing
 *
 * Based on "3D Gaussian Splatting for Real-Time Radiance Field Rendering"
 */
class GaussianRenderer {
public:
    explicit GaussianRenderer(VulkanContext& ctx);
    ~GaussianRenderer();

    // Non-copyable
    GaussianRenderer(const GaussianRenderer&) = delete;
    GaussianRenderer& operator=(const GaussianRenderer&) = delete;

    /**
     * Initialize renderer with given settings
     */
    void init(const RenderSettings& settings);

    /**
     * Resize the render target
     */
    void resize(uint32_t width, uint32_t height);

    /**
     * Upload Gaussian cloud to GPU
     */
    void upload_gaussians(const GaussianCloud& cloud);

    /**
     * Render the scene
     * @param camera The camera for view/projection
     * @return Pointer to RGBA pixel data (width * height * 4 floats)
     */
    const std::vector<float>& render(const Camera& camera);

    /**
     * Get the last rendered image as RGBA bytes [0-255]
     */
    std::vector<uint8_t> get_image_rgba8() const;

    /**
     * Get render statistics
     */
    struct Stats {
        uint32_t total_gaussians;
        uint32_t visible_gaussians;
        uint32_t tiles_x;
        uint32_t tiles_y;
        double project_time_ms;
        double sort_time_ms;
        double render_time_ms;
        double total_time_ms;
    };
    Stats stats() const { return stats_; }

    /**
     * Get current settings
     */
    const RenderSettings& settings() const { return settings_; }

private:
    void compile_shaders();
    void create_buffers();
    void project_gaussians(const Camera& camera);
    void sort_gaussians();
    void rasterize();

    VulkanContext& ctx_;
    ComputePipeline pipeline_;
    RenderSettings settings_;
    Stats stats_;

    // Shader SPIR-V
    std::vector<uint32_t> project_shader_;
    std::vector<uint32_t> sort_shader_;
    std::vector<uint32_t> render_shader_;

    // GPU buffers
    std::shared_ptr<kp::TensorT<float>> gaussians_3d_;      // Input 3D Gaussians
    std::shared_ptr<kp::TensorT<float>> gaussians_2d_;      // Projected 2D Gaussians
    std::shared_ptr<kp::TensorT<uint32_t>> sort_keys_;      // Depth keys for sorting
    std::shared_ptr<kp::TensorT<uint32_t>> sort_indices_;   // Sorted indices
    std::shared_ptr<kp::TensorT<float>> framebuffer_;       // Output RGBA image
    std::shared_ptr<kp::TensorT<float>> camera_data_;       // Camera uniforms

    // CPU-side data
    std::vector<float> pixels_;
    size_t num_gaussians_ = 0;
    bool initialized_ = false;

    // GPU radix sort (replaces CPU sorting for large Gaussian counts)
    std::unique_ptr<GPURadixSort> gpu_sort_;
    static constexpr size_t GPU_SORT_THRESHOLD = 1000;  // Use GPU sort above this count
    static constexpr size_t GPU_SORT_MAX_ELEMENTS = 200000;  // Max supported Gaussians

    void cpu_sort_fallback();  // CPU sorting for small counts or fallback
};

} // namespace fresnel
