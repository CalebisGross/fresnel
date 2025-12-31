#pragma once

#include "image.hpp"
#include "renderer/gaussian.hpp"
#include "renderer/camera.hpp"
#include <glm/glm.hpp>
#include <vector>

namespace fresnel {

/**
 * PointCloud - Collection of 3D points with optional colors
 */
class PointCloud {
public:
    struct Point {
        glm::vec3 position;
        glm::vec3 color;
        float confidence = 1.0f; // Optional: how confident we are about this point
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
     * Convert point cloud to Gaussians for rendering
     *
     * @param point_size Size of each Gaussian (scale)
     * @param opacity Opacity of each Gaussian
     */
    GaussianCloud to_gaussians(float point_size = 0.02f, float opacity = 0.8f) const;

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
