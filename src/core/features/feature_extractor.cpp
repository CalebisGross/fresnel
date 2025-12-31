#include "feature_extractor.hpp"
#include <filesystem>
#include <fstream>
#include <array>
#include <cstdio>
#include <sstream>
#include <algorithm>
#include <cmath>

namespace fresnel {

// Helper to find project root (where .venv and scripts are)
static std::string find_project_root() {
    std::filesystem::path current = std::filesystem::current_path();

    // Walk up looking for scripts/dinov2_inference.py
    for (int i = 0; i < 10; i++) {
        if (std::filesystem::exists(current / "scripts" / "dinov2_inference.py")) {
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

// --- DINOv2Extractor ---

DINOv2Extractor::DINOv2Extractor() {
    std::string root = find_project_root();
    if (!root.empty()) {
        python_path_ = root + "/.venv/bin/python";
        script_path_ = root + "/scripts/dinov2_inference.py";
    }
    model_available_ = check_model_available();
}

bool DINOv2Extractor::check_model_available() {
    if (python_path_.empty() || script_path_.empty()) {
        return false;
    }

    // Check if Python venv exists
    if (!std::filesystem::exists(python_path_)) {
        return false;
    }

    // Check if script exists
    if (!std::filesystem::exists(script_path_)) {
        return false;
    }

    // Check if model exists
    std::string root = find_project_root();
    std::string model_path = root + "/models/dinov2_small.onnx";
    if (!std::filesystem::exists(model_path)) {
        return false;
    }

    return true;
}

FeatureMap DINOv2Extractor::extract(const Image& image) {
    if (image.empty() || !model_available_) {
        return FeatureMap();
    }

    uint32_t w = image.width();
    uint32_t h = image.height();

    // Create temp files for input/output
    std::string temp_dir = std::filesystem::temp_directory_path().string();
    std::string input_path = temp_dir + "/fresnel_dinov2_input.ppm";
    std::string output_path = temp_dir + "/fresnel_dinov2_output.bin";

    // Save input image as PPM (simple format)
    {
        std::ofstream file(input_path, std::ios::binary);
        if (!file) return FeatureMap();

        file << "P6\n" << w << " " << h << "\n255\n";
        for (uint32_t y = 0; y < h; y++) {
            for (uint32_t x = 0; x < w; x++) {
                float r, g, b;
                image.get_rgb(x, y, r, g, b);
                file.put(static_cast<char>(std::clamp(r * 255.0f, 0.0f, 255.0f)));
                file.put(static_cast<char>(std::clamp(g * 255.0f, 0.0f, 255.0f)));
                file.put(static_cast<char>(std::clamp(b * 255.0f, 0.0f, 255.0f)));
            }
        }
    }

    // Run Python inference script
    std::string cmd = python_path_ + " " + script_path_ + " " +
                      input_path + " " + output_path + " 2>&1";

    std::array<char, 256> buffer;
    std::string result;

    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) {
        return FeatureMap();
    }

    while (fgets(buffer.data(), buffer.size(), pipe) != nullptr) {
        result += buffer.data();
    }

    int ret = pclose(pipe);
    if (ret != 0) {
        // Python script failed
        return FeatureMap();
    }

    // Parse output dimensions from stdout: "height width channels"
    uint32_t feat_h = 0, feat_w = 0, feat_c = 0;
    std::istringstream iss(result);
    iss >> feat_h >> feat_w >> feat_c;

    if (feat_h == 0 || feat_w == 0 || feat_c == 0) {
        return FeatureMap();
    }

    // Read output feature map
    size_t num_floats = feat_h * feat_w * feat_c;
    std::vector<float> data(num_floats);

    {
        std::ifstream file(output_path, std::ios::binary);
        if (!file) {
            return FeatureMap();
        }

        file.read(reinterpret_cast<char*>(data.data()), num_floats * sizeof(float));

        if (file.gcount() != static_cast<std::streamsize>(num_floats * sizeof(float))) {
            return FeatureMap();
        }
    }

    // Clean up temp files
    std::filesystem::remove(input_path);
    std::filesystem::remove(output_path);

    return FeatureMap(feat_w, feat_h, feat_c, std::move(data));
}

// --- Factory ---

std::unique_ptr<FeatureExtractor> create_feature_extractor() {
    // Try DINOv2 first
    auto dinov2 = std::make_unique<DINOv2Extractor>();
    if (dinov2->is_ready()) {
        return dinov2;
    }

    // No fallback feature extractor yet
    return nullptr;
}

} // namespace fresnel
