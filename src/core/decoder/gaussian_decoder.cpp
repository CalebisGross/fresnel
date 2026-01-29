#include "gaussian_decoder.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <array>
#include <cstdio>
#include <sstream>
#include <algorithm>
#include <cmath>

namespace fresnel {

// Helper to find project root (where .venv and scripts are)
static std::string find_project_root() {
    std::filesystem::path current = std::filesystem::current_path();

    // Walk up looking for .venv directory
    for (int i = 0; i < 10; i++) {
        if (std::filesystem::exists(current / ".venv")) {
            return current.string();
        }
        if (current.has_parent_path() && current != current.parent_path()) {
            current = current.parent_path();
        } else {
            break;
        }
    }

    return "";
}

// --- LearnedGaussianDecoder ---

LearnedGaussianDecoder::LearnedGaussianDecoder() {
    std::string root = find_project_root();
    if (!root.empty()) {
        python_path_ = root + "/.venv/bin/python";
        script_path_ = root + "/scripts/inference/decoder_inference.py";
    }
    model_available_ = check_model_available();
}

bool LearnedGaussianDecoder::check_model_available() {
    if (python_path_.empty() || script_path_.empty()) {
        std::cerr << "[Decoder] Project root not found\n";
        return false;
    }

    // Check if Python venv exists
    if (!std::filesystem::exists(python_path_)) {
        std::cerr << "[Decoder] Python not found at: " << python_path_ << "\n";
        return false;
    }

    // Check if script exists
    if (!std::filesystem::exists(script_path_)) {
        std::cerr << "[Decoder] Script not found at: " << script_path_ << "\n";
        return false;
    }

    // Check if ONNX model exists
    std::string root = find_project_root();
    std::string model_path = root + "/models/gaussian_decoder.onnx";
    if (!std::filesystem::exists(model_path)) {
        std::cerr << "[Decoder] ONNX model not found at: " << model_path << "\n";
        return false;
    }

    std::cout << "[Decoder] Model available: " << model_path << "\n";
    return true;
}

bool LearnedGaussianDecoder::save_features_temp(const FeatureMap& features, const std::string& path) {
    std::ofstream file(path, std::ios::binary);
    if (!file) return false;

    // Features are stored as HWC, write as raw floats
    // Expected shape: (37, 37, 384)
    file.write(reinterpret_cast<const char*>(features.data()), features.size() * sizeof(float));
    return file.good();
}

bool LearnedGaussianDecoder::save_depth_temp(const DepthMap& depth, const std::string& path) {
    std::ofstream file(path, std::ios::binary);
    if (!file) return false;

    // Save depth as raw floats
    const auto& pixels = depth.pixels();
    file.write(reinterpret_cast<const char*>(pixels.data()), pixels.size() * sizeof(float));
    return file.good();
}

GaussianCloud LearnedGaussianDecoder::decode(const FeatureMap& features, const DepthMap& depth) {
    GaussianCloud cloud;

    if (!model_available_) {
        std::cerr << "[Decoder] Model not available\n";
        return cloud;
    }

    if (features.empty()) {
        std::cerr << "[Decoder] Features empty\n";
        return cloud;
    }

    std::cout << "[Decoder] Running inference with " << features.size() << " feature values\n";
    std::cout << "[Decoder] Depth map size: " << depth.width() << "x" << depth.height() << "\n";

    // Create temp files for input/output
    std::string temp_dir = std::filesystem::temp_directory_path().string();
    std::string features_path = temp_dir + "/fresnel_decoder_features.bin";
    std::string depth_path = temp_dir + "/fresnel_decoder_depth.bin";
    std::string output_path = temp_dir + "/fresnel_decoder_output.bin";

    // Save features to temp file
    if (!save_features_temp(features, features_path)) {
        std::cerr << "[Decoder] Failed to save features to " << features_path << "\n";
        return cloud;
    }

    // Save depth to temp file
    if (!save_depth_temp(depth, depth_path)) {
        std::cerr << "[Decoder] Failed to save depth to " << depth_path << "\n";
        std::filesystem::remove(features_path);
        return cloud;
    }

    // Run Python inference script
    std::string cmd = python_path_ + " " + script_path_ + " " +
                      features_path + " " + depth_path + " " + output_path + " 2>&1";
    std::cout << "[Decoder] Command: " << cmd << "\n";

    std::array<char, 256> buffer;
    std::string result;

    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) {
        std::cerr << "[Decoder] Failed to open pipe for command\n";
        std::filesystem::remove(features_path);
        std::filesystem::remove(depth_path);
        return cloud;
    }

    while (fgets(buffer.data(), buffer.size(), pipe) != nullptr) {
        result += buffer.data();
    }

    int ret = pclose(pipe);
    if (ret != 0) {
        std::cerr << "[Decoder] Python script failed with code " << ret << "\n";
        std::cerr << "[Decoder] Output: " << result << "\n";
        std::filesystem::remove(features_path);
        std::filesystem::remove(depth_path);
        return cloud;
    }

    // Parse number of Gaussians from stdout (first line should be just the count)
    size_t num_gaussians = 0;
    std::istringstream iss(result);
    iss >> num_gaussians;

    std::cout << "[Decoder] Script output: " << result << "\n";
    std::cout << "[Decoder] Parsed num_gaussians: " << num_gaussians << "\n";

    if (num_gaussians == 0) {
        std::cerr << "[Decoder] Got 0 Gaussians from script\n";
        std::filesystem::remove(features_path);
        std::filesystem::remove(depth_path);
        return cloud;
    }

    // Read output Gaussians: 14 floats per Gaussian
    // Layout: position(3), scale(3), rotation(4), color(3), opacity(1)
    const size_t floats_per_gaussian = 14;
    size_t num_floats = num_gaussians * floats_per_gaussian;
    std::vector<float> data(num_floats);

    {
        std::ifstream file(output_path, std::ios::binary);
        if (!file) {
            std::filesystem::remove(features_path);
            std::filesystem::remove(depth_path);
            return cloud;
        }

        file.read(reinterpret_cast<char*>(data.data()), num_floats * sizeof(float));

        if (file.gcount() != static_cast<std::streamsize>(num_floats * sizeof(float))) {
            std::filesystem::remove(features_path);
            std::filesystem::remove(depth_path);
            std::filesystem::remove(output_path);
            return cloud;
        }
    }

    // Convert to GaussianCloud
    cloud.reserve(num_gaussians);

    for (size_t i = 0; i < num_gaussians; i++) {
        const float* g = &data[i * floats_per_gaussian];

        Gaussian3D gaussian;
        gaussian.position = glm::vec3(g[0], g[1], g[2]);
        gaussian.scale = glm::vec3(g[3], g[4], g[5]);
        gaussian.rotation = glm::quat(g[6], g[7], g[8], g[9]);  // w, x, y, z
        gaussian.color = glm::vec3(g[10], g[11], g[12]);
        gaussian.opacity = g[13];

        cloud.add(gaussian);
    }

    // Clean up temp files
    std::filesystem::remove(features_path);
    std::filesystem::remove(depth_path);
    std::filesystem::remove(output_path);

    return cloud;
}

// --- Factory ---

std::unique_ptr<GaussianDecoder> create_gaussian_decoder() {
    auto decoder = std::make_unique<LearnedGaussianDecoder>();
    if (decoder->is_ready()) {
        return decoder;
    }
    return nullptr;
}

} // namespace fresnel
