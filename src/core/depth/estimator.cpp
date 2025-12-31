#include "estimator.hpp"
#include <cmath>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <array>
#include <cstdio>
#include <sstream>

namespace fresnel {

// Helper to find project root (where .venv and scripts are)
static std::string find_project_root() {
    std::filesystem::path current = std::filesystem::current_path();

    // Walk up looking for scripts/depth_inference.py
    for (int i = 0; i < 10; i++) {
        if (std::filesystem::exists(current / "scripts" / "depth_inference.py")) {
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

// --- GradientDepthEstimator ---

DepthMap GradientDepthEstimator::estimate(const Image& image) {
    if (image.empty()) return DepthMap();

    uint32_t w = image.width();
    uint32_t h = image.height();
    DepthMap depth(w, h);

    // Compute gradients (Sobel-like)
    // Higher gradient = edge = likely foreground = closer
    for (uint32_t y = 1; y < h - 1; y++) {
        for (uint32_t x = 1; x < w - 1; x++) {
            float gx = 0.0f, gy = 0.0f;

            // Sample grayscale values
            auto gray = [&](uint32_t px, uint32_t py) {
                float r, g, b;
                image.get_rgb(px, py, r, g, b);
                return 0.299f * r + 0.587f * g + 0.114f * b;
            };

            // Sobel operator
            float tl = gray(x - 1, y - 1), tm = gray(x, y - 1), tr = gray(x + 1, y - 1);
            float ml = gray(x - 1, y),                           mr = gray(x + 1, y);
            float bl = gray(x - 1, y + 1), bm = gray(x, y + 1), br = gray(x + 1, y + 1);

            gx = (tr + 2 * mr + br) - (tl + 2 * ml + bl);
            gy = (bl + 2 * bm + br) - (tl + 2 * tm + tr);

            float gradient = std::sqrt(gx * gx + gy * gy);

            // Invert: high gradient = close (small depth)
            // Add some base depth so flat areas aren't at infinity
            depth.at(x, y) = 1.0f - std::min(gradient * 2.0f, 0.9f);
        }
    }

    // Fill borders
    for (uint32_t x = 0; x < w; x++) {
        depth.at(x, 0) = depth.at(x, 1);
        depth.at(x, h - 1) = depth.at(x, h - 2);
    }
    for (uint32_t y = 0; y < h; y++) {
        depth.at(0, y) = depth.at(1, y);
        depth.at(w - 1, y) = depth.at(w - 2, y);
    }

    // Apply Gaussian blur to smooth the depth map
    DepthMap smoothed(w, h);
    int kernel_size = 5;
    int half = kernel_size / 2;

    for (uint32_t y = 0; y < h; y++) {
        for (uint32_t x = 0; x < w; x++) {
            float sum = 0.0f;
            float weight_sum = 0.0f;

            for (int ky = -half; ky <= half; ky++) {
                for (int kx = -half; kx <= half; kx++) {
                    int px = std::clamp(static_cast<int>(x) + kx, 0, static_cast<int>(w) - 1);
                    int py = std::clamp(static_cast<int>(y) + ky, 0, static_cast<int>(h) - 1);

                    // Gaussian weight
                    float dist = std::sqrt(static_cast<float>(kx * kx + ky * ky));
                    float weight = std::exp(-dist * dist / 2.0f);

                    sum += depth.at(px, py) * weight;
                    weight_sum += weight;
                }
            }

            smoothed.at(x, y) = sum / weight_sum;
        }
    }

    return smoothed;
}

// --- CenterDepthEstimator ---

DepthMap CenterDepthEstimator::estimate(const Image& image) {
    if (image.empty()) return DepthMap();

    uint32_t w = image.width();
    uint32_t h = image.height();
    DepthMap depth(w, h);

    float cx = w * 0.5f;
    float cy = h * 0.5f;
    float max_dist = std::sqrt(cx * cx + cy * cy);

    for (uint32_t y = 0; y < h; y++) {
        for (uint32_t x = 0; x < w; x++) {
            float dx = x - cx;
            float dy = y - cy;
            float dist = std::sqrt(dx * dx + dy * dy) / max_dist;

            // Closer to center = smaller depth (closer)
            // Add some variation based on image intensity
            float r, g, b;
            image.get_rgb(x, y, r, g, b);
            float intensity = 0.299f * r + 0.587f * g + 0.114f * b;

            // Darker pixels might be shadow = further
            // Lighter pixels might be highlighted = closer
            float intensity_factor = 0.2f * (1.0f - intensity);

            depth.at(x, y) = dist * 0.8f + intensity_factor + 0.1f;
        }
    }

    return depth;
}

// --- DepthAnythingEstimator ---

DepthAnythingEstimator::DepthAnythingEstimator() {
    std::string root = find_project_root();
    if (!root.empty()) {
        python_path_ = root + "/.venv/bin/python";
        script_path_ = root + "/scripts/depth_inference.py";
    }
    model_available_ = check_model_available();
}

bool DepthAnythingEstimator::check_model_available() {
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
    std::string model_path = root + "/models/depth_anything_v2_small.onnx";
    if (!std::filesystem::exists(model_path)) {
        return false;
    }

    return true;
}

DepthMap DepthAnythingEstimator::estimate(const Image& image) {
    if (image.empty() || !model_available_) {
        return DepthMap();
    }

    uint32_t w = image.width();
    uint32_t h = image.height();

    // Create temp files for input/output
    std::string temp_dir = std::filesystem::temp_directory_path().string();
    std::string input_path = temp_dir + "/fresnel_depth_input.ppm";
    std::string output_path = temp_dir + "/fresnel_depth_output.bin";

    // Save input image as PPM (simple format)
    {
        std::ofstream file(input_path, std::ios::binary);
        if (!file) return DepthMap();

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
                      input_path + " " + output_path + " " +
                      std::to_string(w) + " " + std::to_string(h) + " 2>&1";

    std::array<char, 256> buffer;
    std::string result;

    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) {
        return DepthMap();
    }

    while (fgets(buffer.data(), buffer.size(), pipe) != nullptr) {
        result += buffer.data();
    }

    int ret = pclose(pipe);
    if (ret != 0) {
        // Python script failed, fall back to placeholder
        return GradientDepthEstimator().estimate(image);
    }

    // Read output depth map
    DepthMap depth(w, h);
    {
        std::ifstream file(output_path, std::ios::binary);
        if (!file) {
            return GradientDepthEstimator().estimate(image);
        }

        file.read(reinterpret_cast<char*>(depth.data()), w * h * sizeof(float));

        if (file.gcount() != static_cast<std::streamsize>(w * h * sizeof(float))) {
            return GradientDepthEstimator().estimate(image);
        }
    }

    // Clean up temp files
    std::filesystem::remove(input_path);
    std::filesystem::remove(output_path);

    return depth;
}

// --- Factory ---

std::unique_ptr<DepthEstimator> create_depth_estimator() {
    // Try Depth Anything V2 first
    auto depth_anything = std::make_unique<DepthAnythingEstimator>();
    if (depth_anything->is_ready()) {
        return depth_anything;
    }

    // Fall back to gradient placeholder
    return std::make_unique<GradientDepthEstimator>();
}

} // namespace fresnel
