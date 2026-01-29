#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <vector>
#include <cstdint>
#include <string>

namespace fresnel {

/**
 * Gaussian3D - A single 3D Gaussian primitive
 *
 * Represents a 3D Gaussian with position, shape (scale + rotation),
 * color, and opacity. This is the fundamental primitive for Gaussian splatting.
 *
 * The covariance matrix Σ is computed from scale (S) and rotation (R) as:
 *   Σ = R * S * S^T * R^T
 *
 * where S is a diagonal matrix of scales.
 */
struct Gaussian3D {
    glm::vec3 position;      // Center position in world space
    glm::vec3 scale;         // Scale along local axes (before rotation)
    glm::quat rotation;      // Orientation as quaternion
    glm::vec3 color;         // RGB color [0,1]
    float opacity;           // Alpha [0,1]

    // Padding for GPU alignment (Gaussians should be 64 bytes aligned)
    float _pad[3];

    Gaussian3D()
        : position(0.0f)
        , scale(1.0f)
        , rotation(glm::identity<glm::quat>())
        , color(1.0f)
        , opacity(1.0f)
        , _pad{0.0f, 0.0f, 0.0f}
    {}

    Gaussian3D(glm::vec3 pos, glm::vec3 scl, glm::quat rot, glm::vec3 col, float alpha)
        : position(pos)
        , scale(scl)
        , rotation(rot)
        , color(col)
        , opacity(alpha)
        , _pad{0.0f, 0.0f, 0.0f}
    {}

    /**
     * Compute the 3x3 covariance matrix from scale and rotation
     */
    glm::mat3 covariance() const {
        glm::mat3 R = glm::mat3_cast(rotation);
        glm::mat3 S = glm::mat3(
            scale.x, 0.0f, 0.0f,
            0.0f, scale.y, 0.0f,
            0.0f, 0.0f, scale.z
        );
        // Σ = R * S * S^T * R^T = R * S^2 * R^T
        glm::mat3 RS = R * S;
        return RS * glm::transpose(RS);
    }
};

// Note: Gaussian3D is manually packed to 16 floats in renderer.cpp for GPU upload
// The struct size doesn't need to exactly match since we serialize manually

/**
 * Gaussian2D - A projected 2D Gaussian for rasterization
 *
 * Result of projecting a 3D Gaussian onto the image plane.
 * Stores the 2D mean, 2D covariance (as inverse for fast evaluation),
 * color, opacity, and depth for sorting.
 */
struct Gaussian2D {
    glm::vec2 mean;          // 2D center on screen (pixels)
    glm::vec3 cov_inv;       // Inverse 2D covariance as (a, b, c) where [[a,b],[b,c]]
    float depth;             // Depth for sorting (view-space z)
    glm::vec3 color;         // RGB color
    float opacity;           // Alpha
    float radius;            // Bounding radius in pixels (for culling)
    uint32_t tile_mask;      // Which tiles this Gaussian overlaps
    float _pad[2];           // Padding to 48 bytes

    Gaussian2D()
        : mean(0.0f)
        , cov_inv(0.0f)
        , depth(0.0f)
        , color(1.0f)
        , opacity(1.0f)
        , radius(0.0f)
        , tile_mask(0)
        , _pad{0.0f, 0.0f}
    {}
};

// Gaussian2D is only used on GPU side (12 floats per Gaussian in shader)

/**
 * GaussianCloud - Collection of 3D Gaussians
 */
class GaussianCloud {
public:
    GaussianCloud() = default;

    void add(const Gaussian3D& g) { gaussians_.push_back(g); }
    void clear() { gaussians_.clear(); }
    void reserve(size_t n) { gaussians_.reserve(n); }

    size_t size() const { return gaussians_.size(); }
    bool empty() const { return gaussians_.empty(); }

    Gaussian3D& operator[](size_t i) { return gaussians_[i]; }
    const Gaussian3D& operator[](size_t i) const { return gaussians_[i]; }

    std::vector<Gaussian3D>& data() { return gaussians_; }
    const std::vector<Gaussian3D>& data() const { return gaussians_; }

    /**
     * Create a test cloud with random Gaussians
     */
    static GaussianCloud create_test_cloud(size_t count, float extent = 5.0f);

    /**
     * Save Gaussians to binary file for Python training.
     *
     * Format: N * 14 floats per Gaussian
     *   - position: 3 floats (x, y, z)
     *   - scale: 3 floats (sx, sy, sz)
     *   - rotation: 4 floats (w, x, y, z) quaternion
     *   - color: 3 floats (r, g, b)
     *   - opacity: 1 float
     *
     * @param path Output file path
     * @return true if successful
     */
    bool save_binary(const std::string& path) const;

    /**
     * Load Gaussians from binary file.
     *
     * @param path Input file path
     * @return true if successful
     */
    bool load_binary(const std::string& path);

    /**
     * Load Gaussians from PLY file (standard 3D Gaussian Splatting format).
     *
     * Reverses the transformations from save_ply:
     * - Scale: exp(log_scale)
     * - Color: f_dc * C0 + 0.5
     * - Opacity: sigmoid(opacity_raw)
     *
     * @param path Input file path (.ply)
     * @return true if successful
     */
    bool load_ply(const std::string& path);

    /**
     * Save Gaussians to PLY file (standard 3D Gaussian Splatting format).
     *
     * Compatible with viewers like:
     * - Blender (with Gaussian Splatting addon)
     * - SuperSplat, Luma AI viewers
     * - MeshLab (as point cloud)
     *
     * @param path Output file path (.ply)
     * @return true if successful
     */
    bool save_ply(const std::string& path) const;

private:
    std::vector<Gaussian3D> gaussians_;
};

} // namespace fresnel
