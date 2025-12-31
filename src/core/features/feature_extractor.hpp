#pragma once

#include "../image.hpp"
#include <memory>
#include <string>
#include <vector>
#include <cstdint>
#include <stdexcept>

namespace fresnel {

/**
 * FeatureMap - 3D tensor of visual features
 *
 * Stores a spatial grid of feature vectors extracted from an image.
 * Layout: (height, width, channels) in row-major order.
 *
 * For DINOv2 with 518x518 input:
 *   - height = width = 37 (patch grid size)
 *   - channels = 384 (feature dimension for DINOv2-Small)
 */
class FeatureMap {
public:
    FeatureMap() : width_(0), height_(0), channels_(0) {}

    FeatureMap(uint32_t width, uint32_t height, uint32_t channels)
        : width_(width), height_(height), channels_(channels),
          data_(width * height * channels, 0.0f) {}

    FeatureMap(uint32_t width, uint32_t height, uint32_t channels, std::vector<float>&& data)
        : width_(width), height_(height), channels_(channels), data_(std::move(data)) {
        if (data_.size() != width * height * channels) {
            throw std::runtime_error("FeatureMap data size mismatch");
        }
    }

    // Dimensions
    uint32_t width() const { return width_; }
    uint32_t height() const { return height_; }
    uint32_t channels() const { return channels_; }
    size_t size() const { return data_.size(); }
    bool empty() const { return data_.empty(); }

    // Access feature vector at (x, y)
    // Returns pointer to the start of the feature vector (channels floats)
    const float* at(uint32_t x, uint32_t y) const {
        return &data_[(y * width_ + x) * channels_];
    }

    float* at(uint32_t x, uint32_t y) {
        return &data_[(y * width_ + x) * channels_];
    }

    // Safe access with bounds checking
    const float* at_safe(uint32_t x, uint32_t y) const {
        if (x >= width_ || y >= height_) {
            throw std::out_of_range("FeatureMap index out of bounds");
        }
        return at(x, y);
    }

    // Access individual element at (x, y, c)
    float get(uint32_t x, uint32_t y, uint32_t c) const {
        return data_[(y * width_ + x) * channels_ + c];
    }

    void set(uint32_t x, uint32_t y, uint32_t c, float value) {
        data_[(y * width_ + x) * channels_ + c] = value;
    }

    // Raw data access
    const float* data() const { return data_.data(); }
    float* data() { return data_.data(); }

    // Get feature vector as a span/view at (x, y)
    std::vector<float> feature_at(uint32_t x, uint32_t y) const {
        const float* ptr = at(x, y);
        return std::vector<float>(ptr, ptr + channels_);
    }

private:
    uint32_t width_;
    uint32_t height_;
    uint32_t channels_;
    std::vector<float> data_;
};


/**
 * FeatureExtractor - Abstract interface for image feature extraction
 *
 * Implementations can use different backends:
 * - ONNX Runtime (for DINOv2, CLIP, etc.)
 * - Vulkan compute shaders
 * - CPU-based feature extraction
 */
class FeatureExtractor {
public:
    virtual ~FeatureExtractor() = default;

    /**
     * Extract features from a single RGB image
     * @param image Input RGB image
     * @return FeatureMap with spatial grid of feature vectors
     */
    virtual FeatureMap extract(const Image& image) = 0;

    /**
     * Get the name of this extractor
     */
    virtual std::string name() const = 0;

    /**
     * Check if this extractor is ready to use
     */
    virtual bool is_ready() const = 0;

    /**
     * Get the feature dimension (channels in output FeatureMap)
     */
    virtual uint32_t feature_dim() const = 0;
};


/**
 * DINOv2Extractor - DINOv2 feature extraction via Python/ONNX
 *
 * Uses the DINOv2 model for rich visual feature extraction.
 * Requires Python environment with onnxruntime and the exported model.
 *
 * Output: 37x37 spatial grid with 384-dim features (for DINOv2-Small)
 */
class DINOv2Extractor : public FeatureExtractor {
public:
    DINOv2Extractor();

    FeatureMap extract(const Image& image) override;
    std::string name() const override { return "DINOv2-Small"; }
    bool is_ready() const override { return model_available_; }
    uint32_t feature_dim() const override { return 384; }

    /**
     * Set custom paths for Python and script
     */
    void set_python_path(const std::string& path) { python_path_ = path; }
    void set_script_path(const std::string& path) { script_path_ = path; }

private:
    std::string python_path_;
    std::string script_path_;
    bool model_available_ = false;

    bool check_model_available();
};


/**
 * Create the best available feature extractor
 * Returns DINOv2 if available, otherwise returns nullptr
 */
std::unique_ptr<FeatureExtractor> create_feature_extractor();

} // namespace fresnel
