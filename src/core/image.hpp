#pragma once

#include <vector>
#include <string>
#include <cstdint>
#include <stdexcept>

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

    // Get min/max depth values
    void get_range(float& min_depth, float& max_depth) const;

    // Normalize to [0,1] range
    DepthMap normalized() const;

    // Convert to RGB visualization
    Image to_colormap() const;

    // Save as grayscale PNG
    bool save(const std::string& path) const;

private:
    uint32_t width_ = 0;
    uint32_t height_ = 0;
    std::vector<float> data_; // Depth values, row-major
};

} // namespace fresnel
