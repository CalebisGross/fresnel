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
 * Create the best available depth estimator
 * Falls back to placeholders if no ML backend available
 */
std::unique_ptr<DepthEstimator> create_depth_estimator();

} // namespace fresnel
