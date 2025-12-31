#pragma once

#include <vector>
#include <string>
#include <cstdint>
#include <stdexcept>
#include <glm/glm.hpp>

namespace fresnel {

/**
 * Image - Simple image container for RGB/RGBA images
 */
class Image {
public:
    Image() = default;
    Image(uint32_t width, uint32_t height, uint32_t channels = 3);

    // Load from file
    static Image load(const std::string& path);

    // Save to file (PNG)
    bool save(const std::string& path) const;

    // Accessors
    uint32_t width() const { return width_; }
    uint32_t height() const { return height_; }
    uint32_t channels() const { return channels_; }
    size_t size() const { return data_.size(); }
    bool empty() const { return data_.empty(); }

    // Data access
    float* data() { return data_.data(); }
    const float* data() const { return data_.data(); }
    std::vector<float>& pixels() { return data_; }
    const std::vector<float>& pixels() const { return data_; }

    // Pixel access (row-major, channels interleaved)
    float& at(uint32_t x, uint32_t y, uint32_t c) {
        return data_[(y * width_ + x) * channels_ + c];
    }
    float at(uint32_t x, uint32_t y, uint32_t c) const {
        return data_[(y * width_ + x) * channels_ + c];
    }

    // Get RGB at pixel
    void get_rgb(uint32_t x, uint32_t y, float& r, float& g, float& b) const {
        size_t idx = (y * width_ + x) * channels_;
        r = data_[idx];
        g = channels_ > 1 ? data_[idx + 1] : r;
        b = channels_ > 2 ? data_[idx + 2] : r;
    }

    // Set RGB at pixel
    void set_rgb(uint32_t x, uint32_t y, float r, float g, float b) {
        size_t idx = (y * width_ + x) * channels_;
        data_[idx] = r;
        if (channels_ > 1) data_[idx + 1] = g;
        if (channels_ > 2) data_[idx + 2] = b;
    }

    // Resize (simple bilinear)
    Image resize(uint32_t new_width, uint32_t new_height) const;

    // Convert to 8-bit for display/saving
    std::vector<uint8_t> to_uint8() const;

private:
    uint32_t width_ = 0;
    uint32_t height_ = 0;
    uint32_t channels_ = 0;
    std::vector<float> data_; // Normalized [0,1], row-major, channels interleaved
};

/**
 * SurfaceInfo - Surface properties computed from depth gradients
 *
 * Used by SAAG (Surface-Aligned Anisotropic Gaussians) to create
 * properly oriented Gaussians that conform to surfaces.
 *
 * Also used by Silhouette Wrapping to detect edges and compute
 * the direction surfaces curve away from camera.
 */
struct SurfaceInfo {
    glm::vec3 normal;      // Surface normal (unit vector)
    float variance;        // Local depth variance (edge indicator)
    float gradient_mag;    // Gradient magnitude (discontinuity detector)
    glm::vec2 gradient_dir; // Normalized gradient direction (points toward deeper)
    float depth_delta;     // Depth difference across the gradient
};

/**
 * DepthMap - Single-channel depth image
 */
class DepthMap {
public:
    DepthMap() = default;
    DepthMap(uint32_t width, uint32_t height);

    uint32_t width() const { return width_; }
    uint32_t height() const { return height_; }
    bool empty() const { return data_.empty(); }

    float* data() { return data_.data(); }
    const float* data() const { return data_.data(); }
    std::vector<float>& pixels() { return data_; }
    const std::vector<float>& pixels() const { return data_; }

    float& at(uint32_t x, uint32_t y) { return data_[y * width_ + x]; }
    float at(uint32_t x, uint32_t y) const { return data_[y * width_ + x]; }

    // Safe access with bounds checking (returns 0 for out-of-bounds)
    float at_safe(int x, int y) const {
        if (x < 0 || x >= static_cast<int>(width_) ||
            y < 0 || y >= static_cast<int>(height_)) {
            return 0.0f;
        }
        return data_[y * width_ + x];
    }

    // Get min/max depth values
    void get_range(float& min_depth, float& max_depth) const;

    // Normalize to [0,1] range
    DepthMap normalized() const;

    // Convert to RGB visualization
    Image to_colormap() const;

    // Save as grayscale PNG
    bool save(const std::string& path) const;

    /**
     * Compute surface information at a pixel using depth gradients
     *
     * Uses Sobel filter to compute depth gradients (dD/dx, dD/dy),
     * then derives surface normal, local variance, and gradient magnitude.
     *
     * @param x Pixel x coordinate
     * @param y Pixel y coordinate
     * @param gradient_scale Scale factor for gradient -> normal conversion
     * @return SurfaceInfo containing normal, variance, and gradient_mag
     */
    SurfaceInfo compute_surface_info(uint32_t x, uint32_t y, float gradient_scale = 1.0f) const;

private:
    uint32_t width_ = 0;
    uint32_t height_ = 0;
    std::vector<float> data_; // Depth values, row-major
};

} // namespace fresnel
