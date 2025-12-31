#include "pointcloud.hpp"
#include <algorithm>
#include <limits>

namespace fresnel {

PointCloud PointCloud::from_depth(
    const DepthMap& depth,
    const Image* color,
    glm::vec4 intrinsics,
    float depth_scale,
    int subsample
) {
    PointCloud cloud;

    if (depth.empty()) return cloud;

    uint32_t w = depth.width();
    uint32_t h = depth.height();

    float fx = intrinsics.x;
    float fy = intrinsics.y;
    float cx = intrinsics.z > 0 ? intrinsics.z : w * 0.5f;
    float cy = intrinsics.w > 0 ? intrinsics.w : h * 0.5f;

    // Normalize depth to get better range
    float min_d, max_d;
    depth.get_range(min_d, max_d);
    float depth_range = max_d - min_d;
    if (depth_range < 1e-6f) depth_range = 1.0f;

    // Reserve approximate capacity
    cloud.reserve((w / subsample) * (h / subsample));

    for (uint32_t y = 0; y < h; y += subsample) {
        for (uint32_t x = 0; x < w; x += subsample) {
            float d = depth.at(x, y);

            // Normalize and scale depth
            // Invert because larger depth values = further away, but in 3D z-positive is toward camera
            float normalized_d = (d - min_d) / depth_range;
            float z = (1.0f - normalized_d) * depth_scale;

            // Skip very far points
            if (z < 0.01f * depth_scale) continue;

            // Unproject: pixel (x, y) with depth z to 3D point
            float X = (x - cx) / fx * z;
            float Y = (cy - y) / fy * z;  // Flip Y for standard 3D coordinates
            float Z = -z;  // Negative because we're looking down -Z

            // Get color
            glm::vec3 col(0.7f, 0.7f, 0.7f); // Default gray
            if (color && !color->empty()) {
                // Handle case where color might be different size
                uint32_t cx_img = std::min(x, color->width() - 1);
                uint32_t cy_img = std::min(y, color->height() - 1);
                color->get_rgb(cx_img, cy_img, col.r, col.g, col.b);
            }

            // Confidence based on depth (closer = more confident)
            float conf = normalized_d;

            cloud.add({glm::vec3(X, Y, Z), col, conf});
        }
    }

    return cloud;
}

GaussianCloud PointCloud::to_gaussians(float point_size, float opacity) const {
    GaussianCloud gaussians;
    gaussians.reserve(points_.size());

    for (const auto& pt : points_) {
        // Scale point size by confidence
        float size = point_size * (0.5f + 0.5f * pt.confidence);

        Gaussian3D g(
            pt.position,
            glm::vec3(size),
            glm::identity<glm::quat>(),
            pt.color,
            opacity * pt.confidence
        );

        gaussians.add(g);
    }

    return gaussians;
}

void PointCloud::get_bounds(glm::vec3& min, glm::vec3& max) const {
    if (points_.empty()) {
        min = max = glm::vec3(0.0f);
        return;
    }

    min = glm::vec3(std::numeric_limits<float>::max());
    max = glm::vec3(std::numeric_limits<float>::lowest());

    for (const auto& pt : points_) {
        min = glm::min(min, pt.position);
        max = glm::max(max, pt.position);
    }
}

void PointCloud::center() {
    if (points_.empty()) return;

    glm::vec3 min, max;
    get_bounds(min, max);
    glm::vec3 center = (min + max) * 0.5f;

    for (auto& pt : points_) {
        pt.position -= center;
    }
}

void PointCloud::normalize(float target_extent) {
    if (points_.empty()) return;

    // First center
    center();

    // Find current extent
    glm::vec3 min, max;
    get_bounds(min, max);
    glm::vec3 extent = max - min;
    float max_extent = std::max({extent.x, extent.y, extent.z});

    if (max_extent < 1e-6f) return;

    // Scale to target extent
    float scale = target_extent / max_extent;
    for (auto& pt : points_) {
        pt.position *= scale;
    }
}

PointCloud create_pointcloud_from_image(
    const Image& image,
    const DepthMap& depth,
    float point_size,
    int subsample
) {
    // Default intrinsics based on image size
    // Assuming ~60 degree FOV
    float f = image.width() * 0.8f;
    glm::vec4 intrinsics(f, f, image.width() * 0.5f, image.height() * 0.5f);

    auto cloud = PointCloud::from_depth(depth, &image, intrinsics, 2.0f, subsample);

    // Normalize to reasonable viewing size
    cloud.normalize(3.0f);

    return cloud;
}

} // namespace fresnel
