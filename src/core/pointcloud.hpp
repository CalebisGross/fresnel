#pragma once

#include "image.hpp"
#include "renderer/gaussian.hpp"
#include "renderer/camera.hpp"
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <vector>

namespace fresnel {

/**
 * SurfaceGaussianParams - Parameters for SAAG (Surface-Aligned Anisotropic Gaussians)
 *
 * Controls how depth gradients are converted to properly oriented Gaussians
 * that conform to surfaces instead of appearing as scattered particles.
 */
struct SurfaceGaussianParams {
    float base_size = 0.008f;       // Base Gaussian size
    float aspect_ratio = 5.0f;      // Surface-tangent / normal ratio (higher = flatter discs)
    float edge_threshold = 0.15f;   // Gradient magnitude threshold for edge detection
    float edge_shrink = 0.3f;       // Shrink factor at edges (0-1)
    float min_confidence = 0.1f;    // Skip points below this confidence
    float gradient_scale = 50.0f;   // Scale factor for gradient -> normal conversion (high for [0,1] depth)
    float normal_strength = 1.0f;   // How much to orient by normal (0 = spheres, 1 = full alignment)
};

/**
 * SilhouetteWrapParams - Parameters for silhouette wrapping
 *
 * At depth discontinuities (silhouettes), surfaces curve away from the camera.
 * This generates extra Gaussians that "wrap around" edges to provide coverage
 * from side views - solving the 2.5D problem.
 */
struct SilhouetteWrapParams {
    bool enabled = true;               // Enable silhouette wrapping
    float edge_threshold = 0.15f;      // Gradient threshold for silhouette detection
    int wrap_layers = 3;               // Number of wrap layers (more = better coverage)
    float layer_spacing = 0.5f;        // Spacing between layers (relative to base_size)
    float opacity_falloff = 0.7f;      // Opacity decay per layer (geometric)
    float max_wrap_angle = 75.0f;      // Maximum angle to wrap (degrees from camera)
    float wrap_aspect = 2.0f;          // Wrap Gaussian aspect (less flat than surface)
};

/**
 * VolumetricShellParams - Parameters for creating 3D volumetric shells
 *
 * Transforms 2.5D surface into true 3D by adding:
 * - Back surface: Gaussians facing away from camera, offset behind front surface
 * - Side walls: At silhouettes, Gaussians connecting front to back
 *
 * This creates a hollow shell with actual thickness, visible from any angle.
 */
struct VolumetricShellParams {
    bool enabled = true;               // Enable volumetric shell
    float thickness = 0.3f;            // Shell thickness (relative to scene scale)
    float back_opacity = 0.6f;         // Back surface opacity multiplier
    float back_darken = 0.8f;          // Darken back surface color (simulates less light)
    bool connect_walls = true;         // Add connecting walls at silhouettes
    int wall_segments = 3;             // Segments between front and back
    float wall_opacity = 0.5f;         // Wall opacity multiplier
    float edge_threshold = 0.1f;       // Only add shell where gradient > threshold (silhouettes)
};

/**
 * AdaptiveDensityParams - Add extra Gaussians at depth edges to fill gaps
 *
 * The pixel grid creates visible stripes when viewed from the side.
 * This adds extra Gaussians with random offsets at depth discontinuities
 * to break the grid pattern and fill gaps.
 */
struct AdaptiveDensityParams {
    bool enabled = true;               // Enable adaptive density
    float gradient_threshold = 0.08f;  // Add density where gradient > this
    int extra_count = 4;               // Extra Gaussians per edge point
    float position_jitter = 0.6f;      // Random offset (relative to gaussian_size)
    float size_variance = 0.3f;        // Size variation (±30%)
    float opacity_scale = 0.7f;        // Extra Gaussian opacity (relative)
    uint32_t seed = 12345;             // RNG seed for reproducibility
};

/**
 * PointCloud - Collection of 3D points with optional colors
 */
class PointCloud {
public:
    struct Point {
        glm::vec3 position;
        glm::vec3 color;
        float confidence = 1.0f; // Optional: how confident we are about this point
        uint32_t pixel_x = 0;    // Source pixel X (for SAAG surface lookup)
        uint32_t pixel_y = 0;    // Source pixel Y (for SAAG surface lookup)
    };

    PointCloud() = default;

    void add(const Point& p) { points_.push_back(p); }
    void add(glm::vec3 pos, glm::vec3 color, float conf = 1.0f) {
        points_.push_back({pos, color, conf});
    }

    void clear() { points_.clear(); }
    void reserve(size_t n) { points_.reserve(n); }

    size_t size() const { return points_.size(); }
    bool empty() const { return points_.empty(); }

    Point& operator[](size_t i) { return points_[i]; }
    const Point& operator[](size_t i) const { return points_[i]; }

    std::vector<Point>& data() { return points_; }
    const std::vector<Point>& data() const { return points_; }

    /**
     * Create point cloud from depth map and color image
     *
     * @param depth Depth map (relative depth values)
     * @param color Optional color image (same size as depth)
     * @param intrinsics Camera intrinsics: (fx, fy, cx, cy)
     * @param depth_scale Scale factor to convert depth to world units
     * @param subsample Sample every Nth pixel (1 = all pixels)
     */
    static PointCloud from_depth(
        const DepthMap& depth,
        const Image* color = nullptr,
        glm::vec4 intrinsics = glm::vec4(500, 500, 0, 0), // fx, fy, cx, cy
        float depth_scale = 1.0f,
        int subsample = 1
    );

    /**
     * Convert point cloud to Gaussians for rendering (legacy - spherical Gaussians)
     *
     * @param point_size Size of each Gaussian (scale)
     * @param opacity Opacity of each Gaussian
     */
    GaussianCloud to_gaussians(float point_size = 0.02f, float opacity = 0.8f) const;

    /**
     * Convert point cloud to SURFACE-ALIGNED ANISOTROPIC Gaussians (SAAG)
     *
     * This is the novel method that creates properly oriented Gaussians
     * that conform to surfaces instead of appearing as scattered particles.
     *
     * Uses depth gradients to compute:
     * - Surface normals for Gaussian orientation
     * - Anisotropic scale (flat discs) aligned to surfaces
     * - Edge detection to shrink Gaussians at depth discontinuities
     * - Silhouette wrapping for side-view coverage (optional)
     * - Volumetric shell for true 3D from any angle (optional)
     * - Adaptive density at edges to break stripe pattern (optional)
     *
     * @param depth The depth map (needed for surface normal computation)
     * @param params SAAG parameters controlling shape, edges, etc.
     * @param wrap_params Silhouette wrapping parameters (optional)
     * @param shell_params Volumetric shell parameters for true 3D (optional)
     * @param density_params Adaptive density to fill gaps at edges (optional)
     * @param opacity Base opacity for Gaussians
     * @return GaussianCloud with surface-aligned anisotropic Gaussians
     */
    GaussianCloud to_surface_gaussians(
        const DepthMap& depth,
        const SurfaceGaussianParams& params = {},
        const SilhouetteWrapParams& wrap_params = {},
        const VolumetricShellParams& shell_params = {},
        const AdaptiveDensityParams& density_params = {},
        float opacity = 0.9f
    ) const;

    /**
     * Get bounding box of the point cloud
     */
    void get_bounds(glm::vec3& min, glm::vec3& max) const;

    /**
     * Center the point cloud around origin
     */
    void center();

    /**
     * Scale the point cloud to fit within a given extent
     */
    void normalize(float target_extent = 2.0f);

private:
    std::vector<Point> points_;
};

/**
 * Helper to create point cloud from image and depth
 *
 * Combines depth estimation and point cloud creation in one step.
 */
PointCloud create_pointcloud_from_image(
    const Image& image,
    const DepthMap& depth,
    float point_size = 0.02f,
    int subsample = 2
);

} // namespace fresnel
