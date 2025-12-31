#pragma once

#include "../image.hpp"
#include <memory>
#include <string>

namespace fresnel {

/**
 * DepthEstimator - Abstract interface for monocular depth estimation
 *
 * Implementations can use different backends:
 * - ONNX Runtime (for Depth Anything V2)
 * - Vulkan compute shaders
 * - Simple gradient-based placeholders
 */
class DepthEstimator {
public:
    virtual ~DepthEstimator() = default;

    /**
     * Estimate depth from a single RGB image
     * @param image Input RGB image
     * @return Depth map with relative depth values (larger = further)
     */
    virtual DepthMap estimate(const Image& image) = 0;

    /**
     * Get the name of this estimator
     */
    virtual std::string name() const = 0;

    /**
     * Check if this estimator is ready to use
     */
    virtual bool is_ready() const = 0;
};

/**
 * GradientDepthEstimator - Simple placeholder depth estimator
 *
 * Creates a depth map based on image gradients (edges = closer).
 * Not accurate, but useful for testing the pipeline.
 */
class GradientDepthEstimator : public DepthEstimator {
public:
    DepthMap estimate(const Image& image) override;
    std::string name() const override { return "Gradient (placeholder)"; }
    bool is_ready() const override { return true; }
};

/**
 * CenterDepthEstimator - Another placeholder
 *
 * Creates depth based on distance from image center.
 * Objects in center appear closer.
 */
class CenterDepthEstimator : public DepthEstimator {
public:
    DepthMap estimate(const Image& image) override;
    std::string name() const override { return "Center-based (placeholder)"; }
    bool is_ready() const override { return true; }
};

/**
 * DepthAnythingEstimator - Depth Anything V2 via Python/ONNX
 *
 * Uses the Depth Anything V2 model for accurate monocular depth estimation.
 * Requires Python environment with onnxruntime and the exported model.
 */
class DepthAnythingEstimator : public DepthEstimator {
public:
    DepthAnythingEstimator();

    DepthMap estimate(const Image& image) override;
    std::string name() const override { return "Depth Anything V2"; }
    bool is_ready() const override { return model_available_; }

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
 * Create the best available depth estimator
 * Returns Depth Anything V2 if available, otherwise falls back to placeholder
 */
std::unique_ptr<DepthEstimator> create_depth_estimator();

} // namespace fresnel
