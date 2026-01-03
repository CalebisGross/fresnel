#pragma once

#include "../renderer/gaussian.hpp"
#include "../features/feature_extractor.hpp"
#include "../image.hpp"
#include <memory>
#include <string>

namespace fresnel {

/**
 * GaussianDecoder - Abstract interface for learned Gaussian prediction
 *
 * Takes visual features and depth, outputs Gaussian primitives.
 * Replaces algorithmic SAAG with learned approach.
 */
class GaussianDecoder {
public:
    virtual ~GaussianDecoder() = default;

    /**
     * Predict Gaussians from features and depth
     *
     * @param features DINOv2 feature map (37x37x384)
     * @param depth Depth map
     * @return GaussianCloud with predicted Gaussians
     */
    virtual GaussianCloud decode(const FeatureMap& features, const DepthMap& depth) = 0;

    /**
     * Get the name of this decoder
     */
    virtual std::string name() const = 0;

    /**
     * Check if this decoder is ready to use
     */
    virtual bool is_ready() const = 0;
};


/**
 * LearnedGaussianDecoder - ONNX-based learned Gaussian prediction
 *
 * Uses trained DirectPatchDecoder model (Experiment 2).
 * Invokes Python subprocess for ONNX inference.
 *
 * Input: DINOv2 features (37x37x384) + depth map
 * Output: ~1369 Gaussians (37x37 grid, 1 per patch)
 */
class LearnedGaussianDecoder : public GaussianDecoder {
public:
    LearnedGaussianDecoder();

    GaussianCloud decode(const FeatureMap& features, const DepthMap& depth) override;
    std::string name() const override { return "DirectPatchDecoder"; }
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

    // Save features to temp file for Python
    bool save_features_temp(const FeatureMap& features, const std::string& path);
    // Save depth to temp file for Python
    bool save_depth_temp(const DepthMap& depth, const std::string& path);
};


/**
 * Create the best available Gaussian decoder
 * Returns learned decoder if available, otherwise nullptr
 */
std::unique_ptr<GaussianDecoder> create_gaussian_decoder();

} // namespace fresnel
