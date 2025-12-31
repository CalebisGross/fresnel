#include "pointcloud.hpp"
#include <algorithm>
#include <limits>
#include <cmath>

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

            Point pt;
            pt.position = glm::vec3(X, Y, Z);
            pt.color = col;
            pt.confidence = conf;
            pt.pixel_x = x;
            pt.pixel_y = y;
            cloud.add(pt);
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

// Helper: compute quaternion that rotates (0,0,1) to align with surface normal
static glm::quat quaternion_from_normal(const glm::vec3& normal) {
    glm::vec3 up(0.0f, 0.0f, 1.0f);
    glm::vec3 axis = glm::cross(up, normal);
    float dot = glm::dot(up, normal);

    float axis_len = glm::length(axis);
    if (axis_len < 1e-6f) {
        // Normal is parallel to up - identity or 180° rotation
        if (dot > 0) {
            return glm::identity<glm::quat>();
        } else {
            return glm::angleAxis(glm::pi<float>(), glm::vec3(1.0f, 0.0f, 0.0f));
        }
    }

    float angle = std::acos(std::clamp(dot, -1.0f, 1.0f));
    return glm::angleAxis(angle, glm::normalize(axis));
}

// Helper: compute the direction to wrap around a silhouette edge
// This is the key innovation for solving the 2.5D problem
static glm::vec3 compute_wrap_direction(
    const glm::vec3& position,
    const glm::vec3& surface_normal,
    const glm::vec2& gradient_dir
) {
    // View direction from camera (assumed at origin looking down -Z)
    glm::vec3 view_dir = glm::normalize(position);

    // Build a coordinate frame aligned with the view
    glm::vec3 world_up(0.0f, 1.0f, 0.0f);
    glm::vec3 right = glm::normalize(glm::cross(world_up, view_dir));
    if (glm::length(right) < 1e-6f) {
        right = glm::vec3(1.0f, 0.0f, 0.0f);
    }
    glm::vec3 up = glm::cross(view_dir, right);

    // Convert 2D gradient direction to 3D (in the image plane)
    glm::vec3 grad_3d = right * gradient_dir.x + up * gradient_dir.y;

    // Wrap direction: perpendicular to both surface normal and gradient
    // This points "around" the silhouette edge
    glm::vec3 wrap = glm::cross(surface_normal, grad_3d);

    // Make sure wrap direction goes away from camera (into the "unseen" side)
    if (glm::dot(wrap, view_dir) < 0) {
        wrap = -wrap;
    }

    float wrap_len = glm::length(wrap);
    if (wrap_len < 1e-6f) {
        // Fallback: use the gradient direction directly
        return glm::normalize(grad_3d);
    }

    return wrap / wrap_len;
}

GaussianCloud PointCloud::to_surface_gaussians(
    const DepthMap& depth,
    const SurfaceGaussianParams& params,
    const SilhouetteWrapParams& wrap_params,
    const VolumetricShellParams& shell_params,
    const AdaptiveDensityParams& density_params,
    float opacity
) const {
    GaussianCloud gaussians;

    // Estimate total Gaussians: base + potential wraps + shell + density
    size_t estimated_size = points_.size();
    if (wrap_params.enabled) {
        // Assume ~20% of points are at silhouettes
        estimated_size += static_cast<size_t>(points_.size() * 0.2f * wrap_params.wrap_layers);
    }
    if (shell_params.enabled) {
        // Back surface doubles the count
        estimated_size *= 2;
        if (shell_params.connect_walls) {
            // Wall segments at silhouettes (~20% of points)
            estimated_size += static_cast<size_t>(points_.size() * 0.2f * shell_params.wall_segments);
        }
    }
    if (density_params.enabled) {
        // Extra Gaussians at edges (~20% of points)
        estimated_size += static_cast<size_t>(points_.size() * 0.2f * density_params.extra_count);
    }
    gaussians.reserve(estimated_size);

    // Deterministic pseudo-random function based on pixel position
    auto pseudo_random = [](uint32_t x, uint32_t y, int i, uint32_t seed) -> float {
        uint32_t h = (x * 374761393u + y * 668265263u + static_cast<uint32_t>(i) * 2147483647u + seed) ^ 0x85ebca6bu;
        h = ((h >> 16) ^ h) * 0x7feb352du;
        return static_cast<float>(h & 0xFFFFu) / 65535.0f;  // 0 to 1
    };

    // Precompute max gradient for normalization
    float max_gradient = 0.0f;
    for (const auto& pt : points_) {
        if (pt.pixel_x < depth.width() && pt.pixel_y < depth.height()) {
            SurfaceInfo info = depth.compute_surface_info(pt.pixel_x, pt.pixel_y, params.gradient_scale);
            max_gradient = std::max(max_gradient, info.gradient_mag);
        }
    }
    if (max_gradient < 1e-6f) max_gradient = 1.0f;

    for (const auto& pt : points_) {
        // Skip low-confidence points
        if (pt.confidence < params.min_confidence) continue;

        // Get surface info at this point's source pixel
        SurfaceInfo info;
        if (pt.pixel_x < depth.width() && pt.pixel_y < depth.height()) {
            info = depth.compute_surface_info(pt.pixel_x, pt.pixel_y, params.gradient_scale);
        } else {
            // Fallback: camera-facing
            info.normal = glm::vec3(0.0f, 0.0f, 1.0f);
            info.gradient_mag = 0.0f;
            info.variance = 0.0f;
            info.gradient_dir = glm::vec2(0.0f, 0.0f);
            info.depth_delta = 0.0f;
        }

        // Compute rotation to align Gaussian with surface
        // Blend between identity (spheres) and full alignment based on normal_strength
        glm::quat surface_rot = quaternion_from_normal(info.normal);
        glm::quat rotation = glm::slerp(glm::identity<glm::quat>(), surface_rot, params.normal_strength);

        // Compute anisotropic scale
        // Surface tangent directions get full size, normal direction gets reduced
        float base = params.base_size * (0.5f + 0.5f * pt.confidence);
        float tangent_scale = base;                           // Full size in surface plane
        float normal_scale = base / params.aspect_ratio;      // Thin perpendicular to surface

        // Edge handling: shrink Gaussians at depth discontinuities
        float normalized_grad = info.gradient_mag / max_gradient;
        float edge_factor = 1.0f;
        if (normalized_grad > params.edge_threshold) {
            // Lerp from 1.0 to edge_shrink as gradient increases
            float t = (normalized_grad - params.edge_threshold) / (1.0f - params.edge_threshold);
            t = std::clamp(t, 0.0f, 1.0f);
            edge_factor = 1.0f - t * (1.0f - params.edge_shrink);
        }

        glm::vec3 scale(
            tangent_scale * edge_factor,
            tangent_scale * edge_factor,
            normal_scale * edge_factor
        );

        // Modulate opacity by confidence and edge factor
        float final_opacity = opacity * pt.confidence * (0.7f + 0.3f * edge_factor);

        Gaussian3D g(
            pt.position,
            scale,
            rotation,
            pt.color,
            final_opacity
        );

        gaussians.add(g);

        // ====== VOLUMETRIC SHELL v2 ======
        // Creates true 3D by adding back surface and connecting walls
        // Only at silhouettes (high gradient) - not everywhere
        // Uses VIEW-ALIGNED offset to prevent divergence on curved surfaces
        glm::vec3 back_pos;  // Declare here for potential use by walls
        if (shell_params.enabled && normalized_grad > shell_params.edge_threshold) {
            // Compute view direction (camera assumed at origin)
            glm::vec3 view_dir = glm::normalize(pt.position);

            // VIEW-ALIGNED OFFSET: Push back along view direction, not surface normal
            // This keeps shell thickness consistent on curved surfaces
            back_pos = pt.position + view_dir * shell_params.thickness;

            // Back surface faces AWAY from camera (toward +view_dir)
            // Compute quaternion that aligns Z with view_dir (facing away from camera)
            glm::quat back_rotation = quaternion_from_normal(view_dir);

            // Back surface is slightly darker (less light reaches it)
            glm::vec3 back_color = pt.color * shell_params.back_darken;

            // Back surface opacity
            float back_opacity_value = final_opacity * shell_params.back_opacity;

            Gaussian3D back_g(
                back_pos,
                scale,  // Same scale as front
                back_rotation,
                back_color,
                back_opacity_value
            );

            gaussians.add(back_g);

            // ====== SIDE WALLS ======
            // Connect front to back with intermediate Gaussians
            if (shell_params.connect_walls) {
                glm::vec3 world_up(0.0f, 1.0f, 0.0f);
                glm::vec3 right = glm::normalize(glm::cross(world_up, view_dir));
                if (glm::length(right) < 1e-6f) {
                    right = glm::vec3(1.0f, 0.0f, 0.0f);
                }
                glm::vec3 up = glm::cross(view_dir, right);

                // Wall tangent follows gradient direction (in image plane)
                glm::vec3 wall_tangent = right * info.gradient_dir.x + up * info.gradient_dir.y;
                float wall_tangent_len = glm::length(wall_tangent);
                if (wall_tangent_len > 0.1f) {
                    wall_tangent = wall_tangent / wall_tangent_len;

                    // Wall normal faces perpendicular to both view and tangent
                    glm::vec3 wall_normal = glm::normalize(glm::cross(view_dir, wall_tangent));

                    // Generate wall segments from front to back
                    for (int seg = 1; seg < shell_params.wall_segments + 1; seg++) {
                        float t = seg / static_cast<float>(shell_params.wall_segments + 1);

                        // Interpolate position from front to back
                        glm::vec3 wall_pos = glm::mix(pt.position, back_pos, t);

                        // Wall faces sideways
                        glm::quat wall_rot = quaternion_from_normal(wall_normal);

                        // Wall opacity - constant for cleaner look
                        float seg_opacity = final_opacity * shell_params.wall_opacity;

                        // Wall scale: similar to front
                        glm::vec3 wall_scale = scale * 0.9f;

                        Gaussian3D wall_g(
                            wall_pos,
                            wall_scale,
                            wall_rot,
                            pt.color,  // Same color
                            seg_opacity
                        );

                        gaussians.add(wall_g);
                    }
                }
            }
        }

        // ====== SILHOUETTE WRAPPING ======
        // At depth discontinuities, add extra Gaussians that face sideways
        // to provide coverage from side views
        if (wrap_params.enabled && normalized_grad > wrap_params.edge_threshold) {
            // This is a silhouette - add wrap Gaussians

            // Only wrap if we have a valid gradient direction
            float grad_dir_len = glm::length(info.gradient_dir);
            if (grad_dir_len > 0.1f) {
                glm::vec3 wrap_dir = compute_wrap_direction(
                    pt.position,
                    info.normal,
                    info.gradient_dir
                );

                // Generate multiple layers of wrap Gaussians
                for (int layer = 0; layer < wrap_params.wrap_layers; layer++) {
                    // Position: offset along wrap direction
                    float offset = (layer + 1) * wrap_params.layer_spacing * params.base_size;
                    glm::vec3 wrap_pos = pt.position + wrap_dir * offset;

                    // Rotation: face perpendicular to wrap direction (toward the "unseen" side)
                    // The wrap Gaussian should face roughly sideways
                    glm::vec3 wrap_normal = -wrap_dir;  // Face back toward visible side
                    glm::quat wrap_rot = quaternion_from_normal(wrap_normal);

                    // Opacity: decay with each layer
                    float wrap_opacity = final_opacity *
                        std::pow(wrap_params.opacity_falloff, static_cast<float>(layer + 1));

                    // Scale: slightly less flat than surface Gaussians (more spherical)
                    float wrap_base = base * 0.8f;  // Slightly smaller
                    glm::vec3 wrap_scale(
                        wrap_base,
                        wrap_base,
                        wrap_base / wrap_params.wrap_aspect
                    );

                    Gaussian3D wrap_g(
                        wrap_pos,
                        wrap_scale,
                        wrap_rot,
                        pt.color,  // Use same color as surface
                        wrap_opacity
                    );

                    gaussians.add(wrap_g);
                }
            }
        }

        // ====== ADAPTIVE DENSITY ======
        // At depth discontinuities, add extra Gaussians with random offsets
        // to break the regular grid pattern that causes stripes from side view
        if (density_params.enabled && normalized_grad > density_params.gradient_threshold) {
            for (int i = 0; i < density_params.extra_count; i++) {
                // Random offset to break grid pattern (deterministic based on pixel + index)
                float rx = (pseudo_random(pt.pixel_x, pt.pixel_y, i * 3 + 0, density_params.seed) - 0.5f) * 2.0f;
                float ry = (pseudo_random(pt.pixel_x, pt.pixel_y, i * 3 + 1, density_params.seed) - 0.5f) * 2.0f;
                float rz = (pseudo_random(pt.pixel_x, pt.pixel_y, i * 3 + 2, density_params.seed) - 0.5f) * 2.0f;

                float jitter = density_params.position_jitter * base;
                glm::vec3 extra_pos = pt.position + glm::vec3(rx, ry, rz) * jitter;

                // Random size variation
                float size_var = 1.0f + (pseudo_random(pt.pixel_x, pt.pixel_y, i * 3 + 100, density_params.seed) - 0.5f)
                    * density_params.size_variance * 2.0f;
                glm::vec3 extra_scale = scale * size_var * 0.8f;

                // Lower opacity for fill Gaussians
                float extra_opacity = final_opacity * density_params.opacity_scale;

                Gaussian3D density_g(
                    extra_pos,
                    extra_scale,
                    rotation,  // Same orientation as parent
                    pt.color,  // Same color
                    extra_opacity
                );

                gaussians.add(density_g);
            }
        }
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
