#include "renderer.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <random>
#include <fstream>
#include <iostream>
#include <sstream>

namespace fresnel {

// Shader source: Project 3D Gaussians to 2D
static const char* PROJECT_SHADER = R"(
#version 450

layout(local_size_x = 256) in;

// Camera uniforms (16 floats)
// [0-3]: view matrix row 0
// [4-7]: view matrix row 1
// [8-11]: view matrix row 2
// [12]: focal_x, [13]: focal_y, [14]: cx, [15]: cy
// [16]: width, [17]: height, [18]: near, [19]: far
layout(binding = 0) buffer CameraData {
    float cam[];
} camera;

// Input: 3D Gaussians (16 floats each = 64 bytes)
// [0-2]: position, [3]: scale.x
// [4-6]: scale.yz, rotation.x, [7]: rotation.y
// Actually let's use simpler layout:
// [0-2]: position, [3-5]: scale, [6-9]: rotation quat, [10-12]: color, [13]: opacity
layout(binding = 1) buffer Gaussians3D {
    float g3d[];
} input_g;

// Output: 2D Gaussians (12 floats each = 48 bytes)
// [0-1]: mean, [2-4]: cov_inv (a,b,c), [5]: depth
// [6-8]: color, [9]: opacity, [10]: radius, [11]: visible flag
layout(binding = 2) buffer Gaussians2D {
    float g2d[];
} output_g;

// Number of Gaussians
layout(binding = 3) buffer GaussianCount {
    uint count;
} params;

mat3 quat_to_mat3(vec4 q) {
    float x = q.x, y = q.y, z = q.z, w = q.w;
    float x2 = x + x, y2 = y + y, z2 = z + z;
    float xx = x * x2, xy = x * y2, xz = x * z2;
    float yy = y * y2, yz = y * z2, zz = z * z2;
    float wx = w * x2, wy = w * y2, wz = w * z2;

    return mat3(
        1.0 - (yy + zz), xy + wz, xz - wy,
        xy - wz, 1.0 - (xx + zz), yz + wx,
        xz + wy, yz - wx, 1.0 - (xx + yy)
    );
}

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= params.count) return;

    // Read 3D Gaussian (16 floats stride)
    uint base3d = idx * 16;
    vec3 pos = vec3(input_g.g3d[base3d], input_g.g3d[base3d + 1], input_g.g3d[base3d + 2]);
    vec3 scale = vec3(input_g.g3d[base3d + 3], input_g.g3d[base3d + 4], input_g.g3d[base3d + 5]);
    vec4 rot = vec4(input_g.g3d[base3d + 6], input_g.g3d[base3d + 7],
                    input_g.g3d[base3d + 8], input_g.g3d[base3d + 9]);
    vec3 color = vec3(input_g.g3d[base3d + 10], input_g.g3d[base3d + 11], input_g.g3d[base3d + 12]);
    float opacity = input_g.g3d[base3d + 13];

    // Read camera view matrix (row-major)
    mat3 view_rot = mat3(
        camera.cam[0], camera.cam[1], camera.cam[2],
        camera.cam[4], camera.cam[5], camera.cam[6],
        camera.cam[8], camera.cam[9], camera.cam[10]
    );
    vec3 view_trans = vec3(camera.cam[3], camera.cam[7], camera.cam[11]);

    float fx = camera.cam[12];
    float fy = camera.cam[13];
    float cx = camera.cam[14];
    float cy = camera.cam[15];
    float width = camera.cam[16];
    float height = camera.cam[17];
    float near_plane = camera.cam[18];

    // Transform position to view space
    vec3 view_pos = view_rot * pos + view_trans;

    // Output base
    uint base2d = idx * 12;

    // Check if behind camera
    if (view_pos.z >= -near_plane) {
        output_g.g2d[base2d + 11] = 0.0; // Not visible
        return;
    }

    float depth = -view_pos.z;

    // Project to screen
    float inv_z = -1.0 / view_pos.z;
    vec2 mean = vec2(
        fx * view_pos.x * inv_z + cx,
        fy * view_pos.y * inv_z + cy
    );

    // Compute 3D covariance: Σ = R * S * S^T * R^T
    mat3 R = quat_to_mat3(rot);
    mat3 S = mat3(
        scale.x * scale.x, 0.0, 0.0,
        0.0, scale.y * scale.y, 0.0,
        0.0, 0.0, scale.z * scale.z
    );
    mat3 cov3d = R * S * transpose(R);

    // Transform covariance to view space
    mat3 cov3d_view = view_rot * cov3d * transpose(view_rot);

    // Compute Jacobian of projection
    // J = | fx/z  0    -fx*x/z^2 |
    //     | 0     fy/z -fy*y/z^2 |
    float z2 = view_pos.z * view_pos.z;
    mat3x2 J = mat3x2(
        fx * inv_z, 0.0,
        0.0, fy * inv_z,
        -fx * view_pos.x / z2, -fy * view_pos.y / z2
    );

    // Project covariance: Σ_2d = J * Σ_view * J^T
    // Note: We take the upper-left 2x2 of the result
    mat2 cov2d = mat2(
        J[0][0] * (cov3d_view[0][0] * J[0][0] + cov3d_view[0][1] * J[1][0] + cov3d_view[0][2] * J[2][0]) +
        J[1][0] * (cov3d_view[1][0] * J[0][0] + cov3d_view[1][1] * J[1][0] + cov3d_view[1][2] * J[2][0]) +
        J[2][0] * (cov3d_view[2][0] * J[0][0] + cov3d_view[2][1] * J[1][0] + cov3d_view[2][2] * J[2][0]),

        J[0][0] * (cov3d_view[0][0] * J[0][1] + cov3d_view[0][1] * J[1][1] + cov3d_view[0][2] * J[2][1]) +
        J[1][0] * (cov3d_view[1][0] * J[0][1] + cov3d_view[1][1] * J[1][1] + cov3d_view[1][2] * J[2][1]) +
        J[2][0] * (cov3d_view[2][0] * J[0][1] + cov3d_view[2][1] * J[1][1] + cov3d_view[2][2] * J[2][1]),

        J[0][1] * (cov3d_view[0][0] * J[0][0] + cov3d_view[0][1] * J[1][0] + cov3d_view[0][2] * J[2][0]) +
        J[1][1] * (cov3d_view[1][0] * J[0][0] + cov3d_view[1][1] * J[1][0] + cov3d_view[1][2] * J[2][0]) +
        J[2][1] * (cov3d_view[2][0] * J[0][0] + cov3d_view[2][1] * J[1][0] + cov3d_view[2][2] * J[2][0]),

        J[0][1] * (cov3d_view[0][0] * J[0][1] + cov3d_view[0][1] * J[1][1] + cov3d_view[0][2] * J[2][1]) +
        J[1][1] * (cov3d_view[1][0] * J[0][1] + cov3d_view[1][1] * J[1][1] + cov3d_view[1][2] * J[2][1]) +
        J[2][1] * (cov3d_view[2][0] * J[0][1] + cov3d_view[2][1] * J[1][1] + cov3d_view[2][2] * J[2][1])
    );

    // Add small epsilon for numerical stability
    cov2d[0][0] += 0.1;
    cov2d[1][1] += 0.1;

    // Invert 2D covariance
    float det = cov2d[0][0] * cov2d[1][1] - cov2d[0][1] * cov2d[1][0];
    if (det <= 0.0) {
        output_g.g2d[base2d + 11] = 0.0; // Not visible
        return;
    }
    float inv_det = 1.0 / det;
    vec3 cov_inv = vec3(
        cov2d[1][1] * inv_det,   // a
        -cov2d[0][1] * inv_det,  // b
        cov2d[0][0] * inv_det    // c
    );

    // Compute bounding radius (3 sigma)
    float eigenvalue = 0.5 * (cov2d[0][0] + cov2d[1][1] +
        sqrt((cov2d[0][0] - cov2d[1][1]) * (cov2d[0][0] - cov2d[1][1]) + 4.0 * cov2d[0][1] * cov2d[0][1]));
    float radius = 3.0 * sqrt(eigenvalue);

    // Frustum culling
    if (mean.x + radius < 0.0 || mean.x - radius > width ||
        mean.y + radius < 0.0 || mean.y - radius > height) {
        output_g.g2d[base2d + 11] = 0.0; // Not visible
        return;
    }

    // Write output
    output_g.g2d[base2d + 0] = mean.x;
    output_g.g2d[base2d + 1] = mean.y;
    output_g.g2d[base2d + 2] = cov_inv.x;
    output_g.g2d[base2d + 3] = cov_inv.y;
    output_g.g2d[base2d + 4] = cov_inv.z;
    output_g.g2d[base2d + 5] = depth;
    output_g.g2d[base2d + 6] = color.x;
    output_g.g2d[base2d + 7] = color.y;
    output_g.g2d[base2d + 8] = color.z;
    output_g.g2d[base2d + 9] = opacity;
    output_g.g2d[base2d + 10] = radius;
    output_g.g2d[base2d + 11] = 1.0; // Visible
}
)";

// Shader source: Simple tile-based rasterization
static const char* RENDER_SHADER = R"(
#version 450

layout(local_size_x = 16, local_size_y = 16) in;

// 2D Gaussians
layout(binding = 0) buffer Gaussians2D {
    float g2d[];
} gaussians;

// Sorted indices
layout(binding = 1) buffer SortedIndices {
    uint indices[];
} sorted;

// Output framebuffer (RGBA)
layout(binding = 2) buffer Framebuffer {
    float pixels[];
} fb;

// Parameters
// [0]: width, [1]: height, [2]: num_gaussians, [3]: bg_r, [4]: bg_g, [5]: bg_b
layout(binding = 3) buffer RenderParams {
    float params[];
} render;

void main() {
    ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
    int width = int(render.params[0]);
    int height = int(render.params[1]);
    uint num_gaussians = uint(render.params[2]);

    if (pixel.x >= width || pixel.y >= height) return;

    vec2 px = vec2(pixel) + 0.5;

    // Initialize with background color
    vec3 color = vec3(render.params[3], render.params[4], render.params[5]);
    float T = 1.0; // Transmittance (remaining opacity)

    // Process Gaussians front-to-back
    for (uint i = 0; i < num_gaussians && T > 0.001; i++) {
        uint idx = sorted.indices[i];
        uint base = idx * 12;

        // Check visibility flag
        if (gaussians.g2d[base + 11] < 0.5) continue;

        vec2 mean = vec2(gaussians.g2d[base + 0], gaussians.g2d[base + 1]);
        float radius = gaussians.g2d[base + 10];

        // Quick reject based on radius
        vec2 diff = px - mean;
        if (abs(diff.x) > radius || abs(diff.y) > radius) continue;

        // Get inverse covariance
        float a = gaussians.g2d[base + 2];
        float b = gaussians.g2d[base + 3];
        float c = gaussians.g2d[base + 4];

        // Evaluate Gaussian: exp(-0.5 * (x^T * Σ^-1 * x))
        float power = -0.5 * (a * diff.x * diff.x + 2.0 * b * diff.x * diff.y + c * diff.y * diff.y);
        if (power < -4.0) continue; // Too far from center

        float alpha = gaussians.g2d[base + 9] * exp(power);
        if (alpha < 0.004) continue; // Below threshold

        vec3 g_color = vec3(gaussians.g2d[base + 6], gaussians.g2d[base + 7], gaussians.g2d[base + 8]);

        // Front-to-back compositing
        color += T * alpha * g_color;
        T *= (1.0 - alpha);
    }

    // Write to framebuffer
    uint fb_idx = (pixel.y * width + pixel.x) * 4;
    fb.pixels[fb_idx + 0] = color.x;
    fb.pixels[fb_idx + 1] = color.y;
    fb.pixels[fb_idx + 2] = color.z;
    fb.pixels[fb_idx + 3] = 1.0 - T; // Alpha
}
)";

GaussianRenderer::GaussianRenderer(VulkanContext& ctx)
    : ctx_(ctx)
    , pipeline_(ctx)
{
}

GaussianRenderer::~GaussianRenderer() = default;

void GaussianRenderer::init(const RenderSettings& settings) {
    settings_ = settings;
    compile_shaders();
    create_buffers();

    // Initialize GPU radix sort
    try {
        gpu_sort_ = std::make_unique<GPURadixSort>(ctx_);
        gpu_sort_->init(GPU_SORT_MAX_ELEMENTS);
    } catch (const std::exception& e) {
        std::cerr << "Warning: GPU radix sort initialization failed: " << e.what()
                  << "\nFalling back to CPU sorting.\n";
        gpu_sort_.reset();
    }

    initialized_ = true;
}

void GaussianRenderer::compile_shaders() {
    project_shader_ = compile_glsl(PROJECT_SHADER);
    render_shader_ = compile_glsl(RENDER_SHADER);
}

void GaussianRenderer::create_buffers() {
    size_t fb_size = settings_.width * settings_.height * 4;
    pixels_.resize(fb_size);

    // Create framebuffer
    framebuffer_ = pipeline_.create_tensor<float>(fb_size);

    // Camera data buffer (20 floats)
    camera_data_ = pipeline_.create_tensor<float>(20);
}

void GaussianRenderer::resize(uint32_t width, uint32_t height) {
    settings_.width = width;
    settings_.height = height;
    create_buffers();
}

void GaussianRenderer::upload_gaussians(const GaussianCloud& cloud) {
    num_gaussians_ = cloud.size();
    if (num_gaussians_ == 0) return;

    // Pack Gaussian data (16 floats per Gaussian)
    std::vector<float> data(num_gaussians_ * 16);
    for (size_t i = 0; i < num_gaussians_; i++) {
        const auto& g = cloud[i];
        size_t base = i * 16;
        data[base + 0] = g.position.x;
        data[base + 1] = g.position.y;
        data[base + 2] = g.position.z;
        data[base + 3] = g.scale.x;
        data[base + 4] = g.scale.y;
        data[base + 5] = g.scale.z;
        data[base + 6] = g.rotation.x;
        data[base + 7] = g.rotation.y;
        data[base + 8] = g.rotation.z;
        data[base + 9] = g.rotation.w;
        data[base + 10] = g.color.r;
        data[base + 11] = g.color.g;
        data[base + 12] = g.color.b;
        data[base + 13] = g.opacity;
        data[base + 14] = 0.0f; // padding
        data[base + 15] = 0.0f; // padding
    }

    gaussians_3d_ = pipeline_.create_tensor<float>(std::span(data));

    // Create 2D projection buffer (12 floats per Gaussian)
    gaussians_2d_ = pipeline_.create_tensor<float>(num_gaussians_ * 12);

    // Create sort buffers
    std::vector<uint32_t> indices(num_gaussians_);
    for (size_t i = 0; i < num_gaussians_; i++) {
        indices[i] = static_cast<uint32_t>(i);
    }
    sort_indices_ = pipeline_.create_tensor<uint32_t>(std::span(indices));
}

void GaussianRenderer::project_gaussians(const Camera& camera) {
    // Pack camera data
    std::vector<float> cam_data(20);

    // View matrix (row-major, 3x4)
    glm::mat4 view = camera.view();
    cam_data[0] = view[0][0]; cam_data[1] = view[1][0]; cam_data[2] = view[2][0]; cam_data[3] = view[3][0];
    cam_data[4] = view[0][1]; cam_data[5] = view[1][1]; cam_data[6] = view[2][1]; cam_data[7] = view[3][1];
    cam_data[8] = view[0][2]; cam_data[9] = view[1][2]; cam_data[10] = view[2][2]; cam_data[11] = view[3][2];

    // Intrinsics
    glm::vec2 focal = camera.focal();
    glm::vec2 pp = camera.principal_point();
    cam_data[12] = focal.x;
    cam_data[13] = focal.y;
    cam_data[14] = pp.x;
    cam_data[15] = pp.y;
    cam_data[16] = static_cast<float>(settings_.width);
    cam_data[17] = static_cast<float>(settings_.height);
    cam_data[18] = camera.near_plane();
    cam_data[19] = camera.far_plane();

    // Update camera buffer
    auto temp_cam = pipeline_.create_tensor<float>(std::span(cam_data));

    // Count buffer
    std::vector<uint32_t> count_data = {static_cast<uint32_t>(num_gaussians_)};
    auto count_buf = pipeline_.create_tensor<uint32_t>(std::span(count_data));

    // Upload and dispatch
    pipeline_.sync_to_device({temp_cam, gaussians_3d_, count_buf});

    pipeline_.dispatch(
        project_shader_,
        {temp_cam, gaussians_3d_, gaussians_2d_, count_buf},
        (num_gaussians_ + 255) / 256
    );
}

void GaussianRenderer::sort_gaussians() {
    // Use GPU radix sort for large Gaussian counts (eliminates CPU-GPU roundtrip)
    if (gpu_sort_ && num_gaussians_ >= GPU_SORT_THRESHOLD && num_gaussians_ <= GPU_SORT_MAX_ELEMENTS) {
        try {
            sort_indices_ = gpu_sort_->sort(gaussians_2d_, num_gaussians_);
            return;
        } catch (const std::exception& e) {
            std::cerr << "GPU sort failed: " << e.what() << ", using CPU fallback\n";
        }
    }

    // Fall back to CPU sorting for small counts or if GPU sort unavailable/failed
    cpu_sort_fallback();
}

void GaussianRenderer::cpu_sort_fallback() {
    // Download 2D Gaussians to get depths
    pipeline_.sync_to_host({gaussians_2d_});
    auto g2d_data = gaussians_2d_->vector();

    // Create depth-index pairs for sorting
    std::vector<std::pair<float, uint32_t>> depth_idx(num_gaussians_);
    for (size_t i = 0; i < num_gaussians_; i++) {
        size_t base = i * 12;
        float visible = g2d_data[base + 11];
        float depth = (visible > 0.5f) ? g2d_data[base + 5] : 1e10f; // Push invisible to back
        depth_idx[i] = {depth, static_cast<uint32_t>(i)};
    }

    // Sort by depth (front-to-back)
    std::sort(depth_idx.begin(), depth_idx.end());

    // Update sort indices
    std::vector<uint32_t> sorted_indices(num_gaussians_);
    for (size_t i = 0; i < num_gaussians_; i++) {
        sorted_indices[i] = depth_idx[i].second;
    }

    sort_indices_ = pipeline_.create_tensor<uint32_t>(std::span(sorted_indices));
    pipeline_.sync_to_device({sort_indices_});
}

void GaussianRenderer::rasterize() {
    // Apply max_render_gaussians limit if set
    size_t render_count = num_gaussians_;
    if (settings_.max_render_gaussians > 0 && settings_.max_render_gaussians < num_gaussians_) {
        render_count = settings_.max_render_gaussians;
    }

    // Render parameters
    std::vector<float> render_params = {
        static_cast<float>(settings_.width),
        static_cast<float>(settings_.height),
        static_cast<float>(render_count),
        settings_.background_color.r,
        settings_.background_color.g,
        settings_.background_color.b
    };
    auto params_buf = pipeline_.create_tensor<float>(std::span(render_params));

    pipeline_.sync_to_device({params_buf, framebuffer_});

    // Dispatch render shader
    uint32_t groups_x = (settings_.width + 15) / 16;
    uint32_t groups_y = (settings_.height + 15) / 16;

    pipeline_.dispatch(
        render_shader_,
        {gaussians_2d_, sort_indices_, framebuffer_, params_buf},
        groups_x, groups_y
    );

    // Download result
    pipeline_.sync_to_host({framebuffer_});
    pixels_ = framebuffer_->vector();
}

const std::vector<float>& GaussianRenderer::render(const Camera& camera) {
    if (!initialized_ || num_gaussians_ == 0) {
        // Return empty/black image
        std::fill(pixels_.begin(), pixels_.end(), 0.0f);
        return pixels_;
    }

    auto t0 = std::chrono::high_resolution_clock::now();

    // 1. Project
    project_gaussians(camera);
    auto t1 = std::chrono::high_resolution_clock::now();

    // 2. Sort
    if (settings_.sort_enabled) {
        sort_gaussians();
    }
    auto t2 = std::chrono::high_resolution_clock::now();

    // 3. Rasterize
    rasterize();
    auto t3 = std::chrono::high_resolution_clock::now();

    // Update stats
    stats_.total_gaussians = num_gaussians_;
    stats_.tiles_x = (settings_.width + settings_.tile_size - 1) / settings_.tile_size;
    stats_.tiles_y = (settings_.height + settings_.tile_size - 1) / settings_.tile_size;
    stats_.project_time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    stats_.sort_time_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
    stats_.render_time_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
    stats_.total_time_ms = std::chrono::duration<double, std::milli>(t3 - t0).count();

    return pixels_;
}

std::vector<uint8_t> GaussianRenderer::get_image_rgba8() const {
    std::vector<uint8_t> result(pixels_.size());
    for (size_t i = 0; i < pixels_.size(); i++) {
        result[i] = static_cast<uint8_t>(std::clamp(pixels_[i] * 255.0f, 0.0f, 255.0f));
    }
    return result;
}

// GaussianCloud implementation
GaussianCloud GaussianCloud::create_test_cloud(size_t count, float extent) {
    GaussianCloud cloud;
    cloud.reserve(count);

    std::mt19937 rng(42); // Fixed seed for reproducibility
    std::uniform_real_distribution<float> pos_dist(-extent, extent);
    std::uniform_real_distribution<float> scale_dist(0.05f, 0.3f);
    std::uniform_real_distribution<float> color_dist(0.2f, 1.0f);
    std::uniform_real_distribution<float> angle_dist(0.0f, glm::two_pi<float>());

    for (size_t i = 0; i < count; i++) {
        glm::vec3 pos(pos_dist(rng), pos_dist(rng), pos_dist(rng));
        glm::vec3 scale(scale_dist(rng), scale_dist(rng), scale_dist(rng));
        glm::vec3 color(color_dist(rng), color_dist(rng), color_dist(rng));

        // Random rotation
        glm::vec3 axis = glm::normalize(glm::vec3(pos_dist(rng), pos_dist(rng), pos_dist(rng)));
        glm::quat rot = glm::angleAxis(angle_dist(rng), axis);

        cloud.add(Gaussian3D(pos, scale, rot, color, 0.8f));
    }

    return cloud;
}

bool GaussianCloud::save_binary(const std::string& path) const {
    std::ofstream file(path, std::ios::binary);
    if (!file) {
        return false;
    }

    // Write each Gaussian as 14 floats
    // Format: position (3) + scale (3) + rotation (4) + color (3) + opacity (1)
    for (const auto& g : gaussians_) {
        // Position
        file.write(reinterpret_cast<const char*>(&g.position.x), sizeof(float));
        file.write(reinterpret_cast<const char*>(&g.position.y), sizeof(float));
        file.write(reinterpret_cast<const char*>(&g.position.z), sizeof(float));

        // Scale
        file.write(reinterpret_cast<const char*>(&g.scale.x), sizeof(float));
        file.write(reinterpret_cast<const char*>(&g.scale.y), sizeof(float));
        file.write(reinterpret_cast<const char*>(&g.scale.z), sizeof(float));

        // Rotation (quaternion w, x, y, z)
        file.write(reinterpret_cast<const char*>(&g.rotation.w), sizeof(float));
        file.write(reinterpret_cast<const char*>(&g.rotation.x), sizeof(float));
        file.write(reinterpret_cast<const char*>(&g.rotation.y), sizeof(float));
        file.write(reinterpret_cast<const char*>(&g.rotation.z), sizeof(float));

        // Color
        file.write(reinterpret_cast<const char*>(&g.color.r), sizeof(float));
        file.write(reinterpret_cast<const char*>(&g.color.g), sizeof(float));
        file.write(reinterpret_cast<const char*>(&g.color.b), sizeof(float));

        // Opacity
        file.write(reinterpret_cast<const char*>(&g.opacity), sizeof(float));
    }

    return file.good();
}

bool GaussianCloud::load_binary(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        return false;
    }

    // Get file size
    auto size = file.tellg();
    file.seekg(0, std::ios::beg);

    // Calculate number of Gaussians (14 floats each)
    size_t num_floats = static_cast<size_t>(size) / sizeof(float);
    size_t num_gaussians = num_floats / 14;

    if (num_gaussians == 0) {
        return false;
    }

    gaussians_.clear();
    gaussians_.reserve(num_gaussians);

    // Read each Gaussian
    for (size_t i = 0; i < num_gaussians; i++) {
        Gaussian3D g;

        // Position
        file.read(reinterpret_cast<char*>(&g.position.x), sizeof(float));
        file.read(reinterpret_cast<char*>(&g.position.y), sizeof(float));
        file.read(reinterpret_cast<char*>(&g.position.z), sizeof(float));

        // Scale
        file.read(reinterpret_cast<char*>(&g.scale.x), sizeof(float));
        file.read(reinterpret_cast<char*>(&g.scale.y), sizeof(float));
        file.read(reinterpret_cast<char*>(&g.scale.z), sizeof(float));

        // Rotation
        file.read(reinterpret_cast<char*>(&g.rotation.w), sizeof(float));
        file.read(reinterpret_cast<char*>(&g.rotation.x), sizeof(float));
        file.read(reinterpret_cast<char*>(&g.rotation.y), sizeof(float));
        file.read(reinterpret_cast<char*>(&g.rotation.z), sizeof(float));

        // Color
        file.read(reinterpret_cast<char*>(&g.color.r), sizeof(float));
        file.read(reinterpret_cast<char*>(&g.color.g), sizeof(float));
        file.read(reinterpret_cast<char*>(&g.color.b), sizeof(float));

        // Opacity
        file.read(reinterpret_cast<char*>(&g.opacity), sizeof(float));

        gaussians_.push_back(g);
    }

    return file.good();
}

bool GaussianCloud::save_ply(const std::string& path) const {
    std::ofstream file(path, std::ios::binary);
    if (!file) {
        return false;
    }

    // Write PLY header (ASCII)
    file << "ply\n";
    file << "format binary_little_endian 1.0\n";
    file << "element vertex " << gaussians_.size() << "\n";
    file << "property float x\n";
    file << "property float y\n";
    file << "property float z\n";
    file << "property float scale_0\n";
    file << "property float scale_1\n";
    file << "property float scale_2\n";
    file << "property float rot_0\n";
    file << "property float rot_1\n";
    file << "property float rot_2\n";
    file << "property float rot_3\n";
    file << "property float f_dc_0\n";
    file << "property float f_dc_1\n";
    file << "property float f_dc_2\n";
    file << "property float opacity\n";
    file << "end_header\n";

    // Write binary data for each Gaussian
    for (const auto& g : gaussians_) {
        // Position (x, y, z)
        file.write(reinterpret_cast<const char*>(&g.position.x), sizeof(float));
        file.write(reinterpret_cast<const char*>(&g.position.y), sizeof(float));
        file.write(reinterpret_cast<const char*>(&g.position.z), sizeof(float));

        // Scale (log-space for compatibility with standard viewers)
        float log_scale_x = std::log(std::max(g.scale.x, 1e-7f));
        float log_scale_y = std::log(std::max(g.scale.y, 1e-7f));
        float log_scale_z = std::log(std::max(g.scale.z, 1e-7f));
        file.write(reinterpret_cast<const char*>(&log_scale_x), sizeof(float));
        file.write(reinterpret_cast<const char*>(&log_scale_y), sizeof(float));
        file.write(reinterpret_cast<const char*>(&log_scale_z), sizeof(float));

        // Rotation (quaternion w, x, y, z)
        file.write(reinterpret_cast<const char*>(&g.rotation.w), sizeof(float));
        file.write(reinterpret_cast<const char*>(&g.rotation.x), sizeof(float));
        file.write(reinterpret_cast<const char*>(&g.rotation.y), sizeof(float));
        file.write(reinterpret_cast<const char*>(&g.rotation.z), sizeof(float));

        // Color (as DC spherical harmonic coefficient, which is color * C0)
        // C0 = 0.28209479177387814 (SH basis for constant term)
        constexpr float C0 = 0.28209479177387814f;
        float f_dc_0 = (g.color.r - 0.5f) / C0;
        float f_dc_1 = (g.color.g - 0.5f) / C0;
        float f_dc_2 = (g.color.b - 0.5f) / C0;
        file.write(reinterpret_cast<const char*>(&f_dc_0), sizeof(float));
        file.write(reinterpret_cast<const char*>(&f_dc_1), sizeof(float));
        file.write(reinterpret_cast<const char*>(&f_dc_2), sizeof(float));

        // Opacity (inverse sigmoid for compatibility)
        float opacity_raw = std::log(g.opacity / std::max(1.0f - g.opacity, 1e-7f));
        file.write(reinterpret_cast<const char*>(&opacity_raw), sizeof(float));
    }

    return file.good();
}

bool GaussianCloud::load_ply(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        return false;
    }

    // Parse PLY header (ASCII)
    std::string line;
    size_t num_vertices = 0;
    bool header_done = false;

    while (std::getline(file, line)) {
        // Remove carriage return if present
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }

        if (line.find("element vertex") != std::string::npos) {
            // Parse: "element vertex 12345"
            std::istringstream iss(line);
            std::string elem, vert;
            iss >> elem >> vert >> num_vertices;
        } else if (line == "end_header") {
            header_done = true;
            break;
        }
    }

    if (!header_done || num_vertices == 0) {
        std::cerr << "Invalid PLY header or no vertices\n";
        return false;
    }

    gaussians_.clear();
    gaussians_.reserve(num_vertices);

    // SH basis constant for DC term
    constexpr float C0 = 0.28209479177387814f;

    // Read binary data
    for (size_t i = 0; i < num_vertices; i++) {
        Gaussian3D g;
        float vals[14];

        file.read(reinterpret_cast<char*>(vals), 14 * sizeof(float));
        if (!file.good()) {
            std::cerr << "Failed reading Gaussian " << i << "\n";
            return false;
        }

        // Position (x, y, z)
        g.position.x = vals[0];
        g.position.y = vals[1];
        g.position.z = vals[2];

        // Scale (exp of log-space values)
        g.scale.x = std::exp(vals[3]);
        g.scale.y = std::exp(vals[4]);
        g.scale.z = std::exp(vals[5]);

        // Rotation (quaternion w, x, y, z)
        g.rotation.w = vals[6];
        g.rotation.x = vals[7];
        g.rotation.y = vals[8];
        g.rotation.z = vals[9];

        // Color (from DC spherical harmonic: color = f_dc * C0 + 0.5)
        g.color.r = std::clamp(vals[10] * C0 + 0.5f, 0.0f, 1.0f);
        g.color.g = std::clamp(vals[11] * C0 + 0.5f, 0.0f, 1.0f);
        g.color.b = std::clamp(vals[12] * C0 + 0.5f, 0.0f, 1.0f);

        // Opacity (sigmoid of raw value)
        g.opacity = 1.0f / (1.0f + std::exp(-vals[13]));

        gaussians_.push_back(g);
    }

    std::cout << "Loaded " << gaussians_.size() << " Gaussians from PLY\n";
    return true;
}

} // namespace fresnel
