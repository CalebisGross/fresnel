#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../../third_party/stb_image.h"
// Note: stb_image_write.h will be added when we need saving

#include "image.hpp"
#include <algorithm>
#include <cmath>

namespace fresnel {

// --- Image ---

Image::Image(uint32_t width, uint32_t height, uint32_t channels)
    : width_(width), height_(height), channels_(channels) {
    data_.resize(width * height * channels, 0.0f);
}

Image Image::load(const std::string& path) {
    int w, h, c;
    unsigned char* raw = stbi_load(path.c_str(), &w, &h, &c, 0);
    if (!raw) {
        throw std::runtime_error("Failed to load image: " + path + " - " + stbi_failure_reason());
    }

    Image img(w, h, c);
    for (size_t i = 0; i < img.data_.size(); i++) {
        img.data_[i] = raw[i] / 255.0f;
    }

    stbi_image_free(raw);
    return img;
}

bool Image::save(const std::string& /*path*/) const {
    // TODO: Implement with stb_image_write
    return false;
}

Image Image::resize(uint32_t new_width, uint32_t new_height) const {
    if (empty()) return Image();

    Image result(new_width, new_height, channels_);

    float x_ratio = static_cast<float>(width_) / new_width;
    float y_ratio = static_cast<float>(height_) / new_height;

    for (uint32_t y = 0; y < new_height; y++) {
        for (uint32_t x = 0; x < new_width; x++) {
            // Bilinear interpolation
            float src_x = x * x_ratio;
            float src_y = y * y_ratio;

            uint32_t x0 = static_cast<uint32_t>(src_x);
            uint32_t y0 = static_cast<uint32_t>(src_y);
            uint32_t x1 = std::min(x0 + 1, width_ - 1);
            uint32_t y1 = std::min(y0 + 1, height_ - 1);

            float fx = src_x - x0;
            float fy = src_y - y0;

            for (uint32_t c = 0; c < channels_; c++) {
                float v00 = at(x0, y0, c);
                float v10 = at(x1, y0, c);
                float v01 = at(x0, y1, c);
                float v11 = at(x1, y1, c);

                float v0 = v00 * (1 - fx) + v10 * fx;
                float v1 = v01 * (1 - fx) + v11 * fx;
                result.at(x, y, c) = v0 * (1 - fy) + v1 * fy;
            }
        }
    }

    return result;
}

std::vector<uint8_t> Image::to_uint8() const {
    std::vector<uint8_t> result(data_.size());
    for (size_t i = 0; i < data_.size(); i++) {
        result[i] = static_cast<uint8_t>(std::clamp(data_[i] * 255.0f, 0.0f, 255.0f));
    }
    return result;
}

// --- DepthMap ---

DepthMap::DepthMap(uint32_t width, uint32_t height)
    : width_(width), height_(height) {
    data_.resize(width * height, 0.0f);
}

void DepthMap::get_range(float& min_depth, float& max_depth) const {
    if (data_.empty()) {
        min_depth = max_depth = 0.0f;
        return;
    }

    min_depth = *std::min_element(data_.begin(), data_.end());
    max_depth = *std::max_element(data_.begin(), data_.end());
}

DepthMap DepthMap::normalized() const {
    DepthMap result(width_, height_);

    float min_d, max_d;
    get_range(min_d, max_d);

    float range = max_d - min_d;
    if (range < 1e-6f) range = 1.0f;

    for (size_t i = 0; i < data_.size(); i++) {
        result.data_[i] = (data_[i] - min_d) / range;
    }

    return result;
}

Image DepthMap::to_colormap() const {
    // Turbo colormap (perceptually uniform)
    auto turbo = [](float t) -> std::tuple<float, float, float> {
        // Simplified turbo colormap
        float r = std::clamp(1.0f - std::abs(t - 0.75f) * 4.0f, 0.0f, 1.0f);
        float g = std::clamp(1.0f - std::abs(t - 0.5f) * 4.0f, 0.0f, 1.0f);
        float b = std::clamp(1.0f - std::abs(t - 0.25f) * 4.0f, 0.0f, 1.0f);

        // Adjust for better visualization
        if (t < 0.25f) {
            b = 0.5f + t * 2.0f;
            r = 0.0f;
        } else if (t > 0.75f) {
            r = 0.5f + (t - 0.75f) * 2.0f;
            b = 0.0f;
        }
        return {r, g, b};
    };

    DepthMap norm = normalized();
    Image result(width_, height_, 3);

    for (uint32_t y = 0; y < height_; y++) {
        for (uint32_t x = 0; x < width_; x++) {
            float d = norm.at(x, y);
            auto [r, g, b] = turbo(d);
            result.set_rgb(x, y, r, g, b);
        }
    }

    return result;
}

bool DepthMap::save(const std::string& /*path*/) const {
    // TODO: Implement with stb_image_write
    return false;
}

SurfaceInfo DepthMap::compute_surface_info(uint32_t x, uint32_t y, float gradient_scale) const {
    SurfaceInfo info;

    // Sobel kernels for gradient computation
    // Gx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    // Gy = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]

    int ix = static_cast<int>(x);
    int iy = static_cast<int>(y);

    // Sample 3x3 neighborhood with safe boundary handling
    float d00 = at_safe(ix - 1, iy - 1);
    float d10 = at_safe(ix,     iy - 1);
    float d20 = at_safe(ix + 1, iy - 1);
    float d01 = at_safe(ix - 1, iy);
    float d11 = at_safe(ix,     iy);      // Center pixel
    float d21 = at_safe(ix + 1, iy);
    float d02 = at_safe(ix - 1, iy + 1);
    float d12 = at_safe(ix,     iy + 1);
    float d22 = at_safe(ix + 1, iy + 1);

    // Compute Sobel gradients
    float gx = (-d00 + d20 - 2.0f * d01 + 2.0f * d21 - d02 + d22) / 8.0f;
    float gy = (-d00 - 2.0f * d10 - d20 + d02 + 2.0f * d12 + d22) / 8.0f;

    // Gradient magnitude (for edge detection)
    info.gradient_mag = std::sqrt(gx * gx + gy * gy);

    // Gradient direction (normalized, points toward deeper regions)
    if (info.gradient_mag > 1e-6f) {
        info.gradient_dir = glm::vec2(gx, gy) / info.gradient_mag;
    } else {
        info.gradient_dir = glm::vec2(0.0f, 0.0f);
    }

    // Depth delta: max depth difference in neighborhood
    float min_d = std::min({d00, d10, d20, d01, d11, d21, d02, d12, d22});
    float max_d = std::max({d00, d10, d20, d01, d11, d21, d02, d12, d22});
    info.depth_delta = max_d - min_d;

    // Compute surface normal from gradients
    // The depth gradient tells us how the surface tilts
    // Normal = normalize(-dD/dx * scale, -dD/dy * scale, 1)
    // Negative because increasing depth = surface tilting away
    glm::vec3 normal(-gx * gradient_scale, -gy * gradient_scale, 1.0f);
    float len = glm::length(normal);
    if (len > 1e-6f) {
        info.normal = normal / len;
    } else {
        info.normal = glm::vec3(0.0f, 0.0f, 1.0f); // Default: facing camera
    }

    // Compute local variance (for edge/confidence detection)
    // Variance of the 3x3 neighborhood
    float mean = (d00 + d10 + d20 + d01 + d11 + d21 + d02 + d12 + d22) / 9.0f;
    float var = 0.0f;
    var += (d00 - mean) * (d00 - mean);
    var += (d10 - mean) * (d10 - mean);
    var += (d20 - mean) * (d20 - mean);
    var += (d01 - mean) * (d01 - mean);
    var += (d11 - mean) * (d11 - mean);
    var += (d21 - mean) * (d21 - mean);
    var += (d02 - mean) * (d02 - mean);
    var += (d12 - mean) * (d12 - mean);
    var += (d22 - mean) * (d22 - mean);
    info.variance = var / 9.0f;

    return info;
}

} // namespace fresnel
