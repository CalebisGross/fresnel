#include "viewer.hpp"

#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <iostream>
#include <fstream>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace fresnel {

Viewer::Viewer() = default;

Viewer::~Viewer() {
    if (window_) {
        shutdown_imgui();
        if (render_texture_) {
            glDeleteTextures(1, &render_texture_);
        }
        glfwDestroyWindow(window_);
        glfwTerminate();
    }
}

void Viewer::glfw_error_callback(int error, const char* description) {
    std::cerr << "GLFW Error " << error << ": " << description << "\n";
}

void Viewer::glfw_scroll_callback(GLFWwindow* window, double /*xoffset*/, double yoffset) {
    auto* viewer = static_cast<Viewer*>(glfwGetWindowUserPointer(window));
    if (viewer && !ImGui::GetIO().WantCaptureMouse) {
        viewer->camera_distance_ *= (1.0f - static_cast<float>(yoffset) * 0.1f);
        viewer->camera_distance_ = std::max(0.5f, std::min(50.0f, viewer->camera_distance_));
        viewer->needs_rerender_ = true;
    }
}

bool Viewer::init(const Settings& settings) {
    settings_ = settings;

    // Initialize GLFW
    glfwSetErrorCallback(glfw_error_callback);
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW\n";
        return false;
    }

    // GL 3.3 + GLSL 330
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // Create window
    window_ = glfwCreateWindow(
        settings_.window_width,
        settings_.window_height,
        settings_.title.c_str(),
        nullptr,
        nullptr
    );

    if (!window_) {
        std::cerr << "Failed to create GLFW window\n";
        glfwTerminate();
        return false;
    }

    glfwMakeContextCurrent(window_);
    glfwSwapInterval(1); // VSync

    // Set up callbacks
    glfwSetWindowUserPointer(window_, this);
    glfwSetScrollCallback(window_, glfw_scroll_callback);

    // Initialize ImGui
    init_imgui();

    // Initialize Vulkan context and renderer
    try {
        vk_context_ = std::make_unique<VulkanContext>();

        RenderSettings render_settings;
        render_settings.width = settings_.render_width;
        render_settings.height = settings_.render_height;
        render_settings.background_color = settings_.background_color;

        renderer_ = std::make_unique<GaussianRenderer>(*vk_context_);
        renderer_->init(render_settings);

        // Initialize camera
        camera_.set_viewport(settings_.render_width, settings_.render_height);
        update_camera();

    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize Vulkan: " << e.what() << "\n";
        return false;
    }

    // Create OpenGL texture for display
    init_render_texture();

    // Initialize depth estimator
    depth_estimator_ = create_depth_estimator();
    stats_.depth_estimator_name = depth_estimator_->name();

    running_ = true;
    return true;
}

void Viewer::init_imgui() {
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    ImGui::StyleColorsDark();

    // Customize style
    ImGuiStyle& style = ImGui::GetStyle();
    style.WindowRounding = 4.0f;
    style.FrameRounding = 2.0f;
    style.Colors[ImGuiCol_WindowBg] = ImVec4(0.1f, 0.1f, 0.12f, 0.95f);

    ImGui_ImplGlfw_InitForOpenGL(window_, true);
    ImGui_ImplOpenGL3_Init("#version 330");
}

void Viewer::shutdown_imgui() {
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
}

void Viewer::init_render_texture() {
    glGenTextures(1, &render_texture_);
    glBindTexture(GL_TEXTURE_2D, render_texture_);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    // Allocate initial texture
    glTexImage2D(
        GL_TEXTURE_2D, 0, GL_RGBA8,
        settings_.render_width, settings_.render_height,
        0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr
    );
}

void Viewer::load_gaussians(const GaussianCloud& cloud) {
    if (renderer_) {
        renderer_->upload_gaussians(cloud);
        stats_.gaussian_count = static_cast<int>(cloud.size());
        needs_rerender_ = true;
    }
}

void Viewer::load_test_cloud(size_t count, float extent) {
    auto cloud = GaussianCloud::create_test_cloud(count, extent);
    load_gaussians(cloud);
}

int Viewer::calculate_auto_subsample() const {
    if (loaded_depth_.empty()) return 1;

    uint32_t w = loaded_depth_.width();
    uint32_t h = loaded_depth_.height();
    size_t total_pixels = static_cast<size_t>(w) * h;

    // Calculate subsample needed to stay under max_gaussians
    int subsample = 1;
    while (total_pixels / (subsample * subsample) > static_cast<size_t>(quality_.max_gaussians)) {
        subsample++;
    }

    return std::max(1, subsample);
}

bool Viewer::load_image(const std::string& path) {
    if (!depth_estimator_) {
        std::cerr << "No depth estimator available\n";
        return false;
    }

    std::cout << "Loading image: " << path << "\n";

    // Load the image
    try {
        loaded_image_ = Image::load(path);
    } catch (const std::exception& e) {
        std::cerr << "Failed to load image: " << e.what() << "\n";
        return false;
    }

    if (loaded_image_.empty()) {
        std::cerr << "Failed to load image (empty)\n";
        return false;
    }

    std::cout << "Image size: " << loaded_image_.width() << "x" << loaded_image_.height() << "\n";
    std::cout << "Running depth estimation with " << depth_estimator_->name() << "...\n";

    // Estimate depth
    loaded_depth_ = depth_estimator_->estimate(loaded_image_);
    if (loaded_depth_.empty()) {
        std::cerr << "Depth estimation failed\n";
        return false;
    }

    std::cout << "Depth map size: " << loaded_depth_.width() << "x" << loaded_depth_.height() << "\n";

    // Update stats
    stats_.loaded_image = path;

    // Process with current quality settings
    reprocess_image();

    // Reset camera for good initial view
    camera_target_ = glm::vec3(0.0f);
    camera_distance_ = 3.5f;
    camera_yaw_ = 0.0f;
    camera_pitch_ = 0.0f;
    needs_rerender_ = true;

    std::cout << "Image loaded successfully!\n";
    return true;
}

void Viewer::reprocess_image(bool preview) {
    if (loaded_image_.empty() || loaded_depth_.empty()) {
        return;
    }

    uint32_t w = loaded_depth_.width();
    uint32_t h = loaded_depth_.height();

    // Determine subsample rate
    int subsample = quality_.subsample;
    float splat_size = quality_.gaussian_size;

    if (preview) {
        // Use faster preview settings
        int auto_sub = calculate_auto_subsample();
        subsample = std::max(subsample, std::max(auto_sub, preview_subsample_));
        // Slightly larger splats for preview to fill gaps
        splat_size = quality_.gaussian_size * (static_cast<float>(subsample) / std::max(1, quality_.subsample));
    } else if (quality_.auto_quality) {
        // Auto-adjust to stay under max_gaussians
        int auto_sub = calculate_auto_subsample();
        if (auto_sub > subsample) {
            subsample = auto_sub;
            std::cout << "Auto-adjusted subsample to " << subsample
                      << " (max " << quality_.max_gaussians << " Gaussians)\n";
        }
    }

    // Apply depth curve transformation for better visual depth
    DepthMap adjusted_depth(w, h);
    for (uint32_t y = 0; y < h; y++) {
        for (uint32_t x = 0; x < w; x++) {
            float d = loaded_depth_.at(x, y);
            adjusted_depth.at(x, y) = std::pow(d, quality_.depth_exponent);
        }
    }

    // Create camera intrinsics based on image size
    float fx = static_cast<float>(w) * 0.8f;
    float fy = fx;
    float cx = w * 0.5f;
    float cy = h * 0.5f;
    glm::vec4 intrinsics(fx, fy, cx, cy);

    // Create point cloud
    auto pointcloud = PointCloud::from_depth(
        adjusted_depth,
        &loaded_image_,
        intrinsics,
        quality_.depth_scale,
        subsample
    );

    if (pointcloud.empty()) {
        std::cerr << "Point cloud creation failed\n";
        return;
    }

    // Center and normalize
    pointcloud.center();
    pointcloud.normalize(3.0f);

    // Convert to Gaussians
    GaussianCloud gaussians;

    if (quality_.use_saag) {
        // Use SAAG: Surface-Aligned Anisotropic Gaussians
        SurfaceGaussianParams params;
        params.base_size = splat_size;
        params.aspect_ratio = quality_.aspect_ratio;
        params.edge_threshold = quality_.edge_threshold;
        params.edge_shrink = quality_.edge_shrink;
        params.gradient_scale = quality_.gradient_scale;
        params.normal_strength = quality_.normal_strength;
        params.min_confidence = 0.1f;

        // Silhouette Wrapping params (solves 2.5D problem)
        SilhouetteWrapParams wrap_params;
        wrap_params.enabled = quality_.silhouette_wrap;
        wrap_params.wrap_layers = quality_.wrap_layers;
        wrap_params.layer_spacing = quality_.wrap_spacing;
        wrap_params.opacity_falloff = quality_.wrap_opacity;
        wrap_params.edge_threshold = quality_.wrap_edge_threshold;

        // Volumetric Shell params (true 3D from any angle)
        VolumetricShellParams shell_params;
        shell_params.enabled = quality_.volumetric_shell;
        shell_params.thickness = quality_.shell_thickness;
        shell_params.back_opacity = quality_.back_opacity;
        shell_params.back_darken = quality_.back_darken;
        shell_params.connect_walls = quality_.connect_walls;
        shell_params.wall_segments = quality_.wall_segments;
        shell_params.wall_opacity = quality_.wall_opacity;
        shell_params.edge_threshold = quality_.shell_edge_threshold;

        // Adaptive Density params (breaks stripe pattern at edges)
        AdaptiveDensityParams density_params;
        density_params.enabled = quality_.adaptive_density;
        density_params.gradient_threshold = quality_.density_threshold;
        density_params.extra_count = quality_.density_extra;
        density_params.position_jitter = quality_.density_jitter;

        gaussians = pointcloud.to_surface_gaussians(adjusted_depth, params, wrap_params, shell_params, density_params, quality_.opacity);

        if (!preview) {
            std::cout << "Generated " << gaussians.size() << " Gaussians"
                      << " (SAAG"
                      << (quality_.silhouette_wrap ? "+Wrap" : "")
                      << (quality_.volumetric_shell ? "+Shell" : "")
                      << (quality_.adaptive_density ? "+Density" : "")
                      << ", subsample=" << subsample << ")\n";
        }
    } else {
        // Legacy: spherical Gaussians
        gaussians = pointcloud.to_gaussians(splat_size, quality_.opacity);

        if (!preview) {
            std::cout << "Generated " << gaussians.size() << " Gaussians"
                      << " (subsample=" << subsample << ")\n";
        }
    }

    // Upload to renderer
    load_gaussians(gaussians);
    needs_rerender_ = true;
}

bool Viewer::export_image(const std::string& path) {
    if (loaded_image_.empty() || loaded_depth_.empty()) {
        std::cerr << "No image loaded to export\n";
        return false;
    }

    std::cout << "Generating high-quality export...\n";

    // Temporarily disable auto-quality for maximum resolution
    bool old_auto = quality_.auto_quality;
    int old_max = quality_.max_gaussians;
    quality_.auto_quality = false;
    quality_.max_gaussians = 10000000; // 10M max for export

    // Reprocess at full quality
    reprocess_image(false);

    // Render
    update_camera();
    renderer_->render(camera_);

    // Get the rendered image
    auto rgba8 = renderer_->get_image_rgba8();

    // Determine output format based on extension
    std::string ppm_path = path;
    bool convert_to_png = false;

    if (path.size() > 4) {
        std::string ext = path.substr(path.size() - 4);
        if (ext == ".png" || ext == ".PNG") {
            // Save as PPM first, then convert
            ppm_path = path.substr(0, path.size() - 4) + ".ppm";
            convert_to_png = true;
        }
    }

    // Save as PPM
    {
        std::ofstream file(ppm_path, std::ios::binary);
        if (!file) {
            std::cerr << "Failed to open file for export: " << ppm_path << "\n";
            quality_.auto_quality = old_auto;
            quality_.max_gaussians = old_max;
            reprocess_image(false);
            return false;
        }

        file << "P6\n" << settings_.render_width << " " << settings_.render_height << "\n255\n";
        for (int i = 0; i < settings_.render_width * settings_.render_height; i++) {
            file.put(static_cast<char>(rgba8[i * 4 + 0]));
            file.put(static_cast<char>(rgba8[i * 4 + 1]));
            file.put(static_cast<char>(rgba8[i * 4 + 2]));
        }
    }

    // Convert to PNG if requested
    if (convert_to_png) {
        std::string cmd = "convert " + ppm_path + " " + path + " 2>/dev/null";
        int ret = system(cmd.c_str());
        if (ret == 0) {
            // Remove temp PPM file
            std::remove(ppm_path.c_str());
            std::cout << "Exported PNG to: " << path << "\n";
        } else {
            std::cout << "ImageMagick not found. Saved PPM to: " << ppm_path << "\n";
            std::cout << "Convert manually: convert " << ppm_path << " " << path << "\n";
        }
    } else {
        std::cout << "Exported to: " << path << "\n";
    }

    // Restore settings and reprocess for preview
    quality_.auto_quality = old_auto;
    quality_.max_gaussians = old_max;
    reprocess_image(false);

    return true;
}

void Viewer::update_camera() {
    // Spherical to Cartesian
    float x = camera_distance_ * std::cos(camera_pitch_) * std::sin(camera_yaw_);
    float y = camera_distance_ * std::sin(camera_pitch_);
    float z = camera_distance_ * std::cos(camera_pitch_) * std::cos(camera_yaw_);

    glm::vec3 camera_pos = camera_target_ + glm::vec3(x, y, z);
    camera_.look_at(camera_pos, camera_target_);
}

void Viewer::process_input() {
    ImGuiIO& io = ImGui::GetIO();

    // Only process mouse if ImGui doesn't want it
    if (!io.WantCaptureMouse) {
        double mouse_x, mouse_y;
        glfwGetCursorPos(window_, &mouse_x, &mouse_y);

        // Left mouse button: orbit
        if (glfwGetMouseButton(window_, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS) {
            if (!mouse_dragging_) {
                mouse_dragging_ = true;
                last_mouse_x_ = mouse_x;
                last_mouse_y_ = mouse_y;
            } else {
                double dx = mouse_x - last_mouse_x_;
                double dy = mouse_y - last_mouse_y_;

                camera_yaw_ -= static_cast<float>(dx) * 0.01f;
                camera_pitch_ += static_cast<float>(dy) * 0.01f;

                // Clamp pitch
                camera_pitch_ = std::max(-1.5f, std::min(1.5f, camera_pitch_));

                last_mouse_x_ = mouse_x;
                last_mouse_y_ = mouse_y;
                needs_rerender_ = true;
            }
        } else {
            mouse_dragging_ = false;
        }

        // Right mouse button: pan
        if (glfwGetMouseButton(window_, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS) {
            if (!mouse_dragging_) {
                mouse_dragging_ = true;
                last_mouse_x_ = mouse_x;
                last_mouse_y_ = mouse_y;
            } else {
                double dx = mouse_x - last_mouse_x_;
                double dy = mouse_y - last_mouse_y_;

                // Pan in camera-local coordinates
                glm::vec3 right = glm::normalize(glm::cross(
                    camera_target_ - camera_.position(),
                    glm::vec3(0, 1, 0)
                ));
                glm::vec3 up = glm::vec3(0, 1, 0);

                float pan_speed = camera_distance_ * 0.002f;
                camera_target_ -= right * static_cast<float>(dx) * pan_speed;
                camera_target_ += up * static_cast<float>(dy) * pan_speed;

                last_mouse_x_ = mouse_x;
                last_mouse_y_ = mouse_y;
                needs_rerender_ = true;
            }
        } else if (glfwGetMouseButton(window_, GLFW_MOUSE_BUTTON_LEFT) != GLFW_PRESS) {
            mouse_dragging_ = false;
        }
    }

    // Keyboard shortcuts
    if (glfwGetKey(window_, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
        close();
    }

    if (glfwGetKey(window_, GLFW_KEY_R) == GLFW_PRESS) {
        // Reset camera
        camera_target_ = glm::vec3(0.0f);
        camera_distance_ = 5.0f;
        camera_yaw_ = 0.0f;
        camera_pitch_ = 0.3f;
        needs_rerender_ = true;
    }
}

void Viewer::render_frame() {
    if (!needs_rerender_ || !renderer_) return;

    update_camera();

    auto start = std::chrono::high_resolution_clock::now();

    // Render Gaussians
    renderer_->render(camera_);

    auto end = std::chrono::high_resolution_clock::now();
    stats_.render_time_ms = std::chrono::duration<float, std::milli>(end - start).count();

    // Get rendered pixels and upload to OpenGL texture
    auto rgba8 = renderer_->get_image_rgba8();

    glBindTexture(GL_TEXTURE_2D, render_texture_);
    glTexSubImage2D(
        GL_TEXTURE_2D, 0, 0, 0,
        settings_.render_width, settings_.render_height,
        GL_RGBA, GL_UNSIGNED_BYTE, rgba8.data()
    );

    needs_rerender_ = false;
}

void Viewer::render_ui() {
    // Main viewport window
    ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(
        ImVec2(static_cast<float>(settings_.render_width) + 20,
               static_cast<float>(settings_.render_height) + 40),
        ImGuiCond_FirstUseEver
    );

    if (ImGui::Begin("Viewport", nullptr, ImGuiWindowFlags_NoScrollbar)) {
        ImVec2 avail = ImGui::GetContentRegionAvail();
        float aspect = static_cast<float>(settings_.render_width) / settings_.render_height;

        float img_w = avail.x;
        float img_h = img_w / aspect;
        if (img_h > avail.y) {
            img_h = avail.y;
            img_w = img_h * aspect;
        }

        // Center the image
        ImVec2 pos = ImGui::GetCursorPos();
        pos.x += (avail.x - img_w) * 0.5f;
        pos.y += (avail.y - img_h) * 0.5f;
        ImGui::SetCursorPos(pos);

        ImGui::Image(
            (ImTextureID)(intptr_t)render_texture_,
            ImVec2(img_w, img_h),
            ImVec2(0, 1), ImVec2(1, 0) // Flip Y for OpenGL
        );
    }
    ImGui::End();

    // Stats panel
    ImGui::SetNextWindowPos(
        ImVec2(static_cast<float>(settings_.render_width) + 40, 10),
        ImGuiCond_FirstUseEver
    );
    ImGui::SetNextWindowSize(ImVec2(250, 200), ImGuiCond_FirstUseEver);

    if (ImGui::Begin("Stats")) {
        ImGui::Text("FPS: %.1f", stats_.fps);
        ImGui::Text("Frame Time: %.2f ms", stats_.frame_time_ms);
        ImGui::Text("Render Time: %.2f ms", stats_.render_time_ms);
        ImGui::Separator();
        ImGui::Text("Gaussians: %d", stats_.gaussian_count);
        ImGui::Separator();
        ImGui::Text("Depth: %s", stats_.depth_estimator_name.c_str());
        ImGui::Separator();
        ImGui::Text("GPU: %s", vk_context_->device_info().name.c_str());
    }
    ImGui::End();

    // Controls panel
    ImGui::SetNextWindowPos(
        ImVec2(static_cast<float>(settings_.render_width) + 40, 220),
        ImGuiCond_FirstUseEver
    );
    ImGui::SetNextWindowSize(ImVec2(250, 450), ImGuiCond_FirstUseEver);

    if (ImGui::Begin("Controls")) {
        // Image loading section
        ImGui::Text("Load Image:");
        ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
        ImGui::InputText("##imagepath", image_path_buffer_, sizeof(image_path_buffer_));

        if (ImGui::Button("Load Image", ImVec2(-1, 0))) {
            if (image_path_buffer_[0] != '\0') {
                load_image(image_path_buffer_);
            }
        }

        if (!stats_.loaded_image.empty()) {
            // Extract just the filename for display
            std::string filename = stats_.loaded_image;
            size_t last_slash = filename.find_last_of("/\\");
            if (last_slash != std::string::npos) {
                filename = filename.substr(last_slash + 1);
            }
            ImGui::TextWrapped("Loaded: %s", filename.c_str());
        }

        // Quality settings section
        if (!loaded_image_.empty()) {
            ImGui::Separator();
            ImGui::Text("Quality Settings:");

            bool slider_active = false;
            bool slider_changed = false;

            // Resolution (subsample)
            const char* res_items[] = { "Full", "1/2", "1/4", "1/8" };
            int res_idx = 0;
            if (quality_.subsample == 2) res_idx = 1;
            else if (quality_.subsample == 4) res_idx = 2;
            else if (quality_.subsample >= 8) res_idx = 3;

            if (ImGui::Combo("Resolution", &res_idx, res_items, 4)) {
                int new_subsample = (res_idx == 0) ? 1 : (res_idx == 1) ? 2 : (res_idx == 2) ? 4 : 8;
                if (new_subsample != quality_.subsample) {
                    quality_.subsample = new_subsample;
                    reprocess_image(false); // Immediate for combo
                }
            }

            // Sliders use preview mode while dragging
            if (ImGui::SliderFloat("Splat Size", &quality_.gaussian_size, 0.002f, 0.05f, "%.3f")) {
                slider_changed = true;
            }
            slider_active |= ImGui::IsItemActive();

            if (ImGui::SliderFloat("Opacity", &quality_.opacity, 0.5f, 1.0f, "%.2f")) {
                slider_changed = true;
            }
            slider_active |= ImGui::IsItemActive();

            if (ImGui::SliderFloat("Depth Scale", &quality_.depth_scale, 0.5f, 5.0f, "%.1f")) {
                slider_changed = true;
            }
            slider_active |= ImGui::IsItemActive();

            if (ImGui::SliderFloat("Depth Curve", &quality_.depth_exponent, 0.3f, 2.0f, "%.2f")) {
                slider_changed = true;
            }
            slider_active |= ImGui::IsItemActive();
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("< 1.0 = more foreground detail\n> 1.0 = more background detail");
            }

            ImGui::Separator();
            ImGui::Text("Performance:");

            // Auto quality toggle
            if (ImGui::Checkbox("Auto Quality", &quality_.auto_quality)) {
                reprocess_image(false);
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Automatically limit point count for smooth performance");
            }

            // Max gaussians slider (only if auto quality on)
            if (quality_.auto_quality) {
                int max_k = quality_.max_gaussians / 1000;
                if (ImGui::SliderInt("Max Points (K)", &max_k, 100, 2000)) {
                    quality_.max_gaussians = max_k * 1000;
                    reprocess_image(false);
                }
            }

            // SAAG settings
            ImGui::Separator();
            ImGui::Text("Surface Alignment (SAAG):");

            if (ImGui::Checkbox("Enable SAAG", &quality_.use_saag)) {
                reprocess_image(false);
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Surface-Aligned Anisotropic Gaussians\nCreates flat disc-like Gaussians that\nconform to surfaces instead of spheres");
            }

            if (quality_.use_saag) {
                // SAAG-specific sliders
                if (ImGui::SliderFloat("Flatness", &quality_.aspect_ratio, 1.0f, 15.0f, "%.1f")) {
                    slider_changed = true;
                }
                slider_active |= ImGui::IsItemActive();
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("How flat the Gaussians are\n1 = spheres, higher = flatter discs");
                }

                if (ImGui::SliderFloat("Normal Strength", &quality_.normal_strength, 0.0f, 1.0f, "%.2f")) {
                    slider_changed = true;
                }
                slider_active |= ImGui::IsItemActive();
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("How much to align with surface\n0 = no alignment (spheres)\n1 = full alignment");
                }

                if (ImGui::SliderFloat("Edge Threshold", &quality_.edge_threshold, 0.05f, 0.5f, "%.2f")) {
                    slider_changed = true;
                }
                slider_active |= ImGui::IsItemActive();
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Sensitivity to depth discontinuities\nLower = more edge detection");
                }

                if (ImGui::SliderFloat("Edge Shrink", &quality_.edge_shrink, 0.1f, 1.0f, "%.2f")) {
                    slider_changed = true;
                }
                slider_active |= ImGui::IsItemActive();
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("How much to shrink at edges\nLower = smaller Gaussians at depth edges");
                }

                if (ImGui::SliderFloat("Gradient Scale", &quality_.gradient_scale, 10.0f, 150.0f, "%.0f")) {
                    slider_changed = true;
                }
                slider_active |= ImGui::IsItemActive();
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Sensitivity of normal estimation\nHigher = more surface detail");
                }
            }

            // Silhouette Wrapping - solves 2.5D problem
            ImGui::Separator();
            ImGui::Text("Silhouette Wrapping:");
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Adds Gaussians at edges for side-view coverage\nSolves the '2.5D' problem");
            }

            if (ImGui::Checkbox("Enable Wrap", &quality_.silhouette_wrap)) {
                slider_changed = true;
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Adds extra Gaussians at silhouettes\nImproves appearance from side angles");
            }

            if (quality_.silhouette_wrap) {
                if (ImGui::SliderInt("Wrap Layers", &quality_.wrap_layers, 1, 6)) {
                    slider_changed = true;
                }
                slider_active |= ImGui::IsItemActive();
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Number of wrap layers\nMore = better side coverage, more Gaussians");
                }

                if (ImGui::SliderFloat("Wrap Spacing", &quality_.wrap_spacing, 0.2f, 1.5f, "%.2f")) {
                    slider_changed = true;
                }
                slider_active |= ImGui::IsItemActive();
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Distance between wrap layers\nHigher = more spread out");
                }

                if (ImGui::SliderFloat("Wrap Opacity", &quality_.wrap_opacity, 0.3f, 0.95f, "%.2f")) {
                    slider_changed = true;
                }
                slider_active |= ImGui::IsItemActive();
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Opacity falloff per layer\nHigher = more visible wrap layers");
                }

                if (ImGui::SliderFloat("Wrap Threshold", &quality_.wrap_edge_threshold, 0.05f, 0.4f, "%.2f")) {
                    slider_changed = true;
                }
                slider_active |= ImGui::IsItemActive();
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Gradient threshold for silhouette detection\nLower = more edges wrapped");
                }
            }

            // Volumetric Shell - true 3D from any angle
            ImGui::Separator();
            ImGui::Text("Volumetric Shell:");
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Creates true 3D by adding back surface and walls\nLook solid from ANY viewing angle");
            }

            if (ImGui::Checkbox("Enable Shell", &quality_.volumetric_shell)) {
                slider_changed = true;
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Add back surface and side walls\nTransforms 2.5D surface into hollow 3D shell");
            }

            if (quality_.volumetric_shell) {
                if (ImGui::SliderFloat("Shell Thickness", &quality_.shell_thickness, 0.1f, 1.0f, "%.2f")) {
                    slider_changed = true;
                }
                slider_active |= ImGui::IsItemActive();
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Distance between front and back surfaces\nHigher = thicker shell");
                }

                if (ImGui::SliderFloat("Back Opacity", &quality_.back_opacity, 0.2f, 0.9f, "%.2f")) {
                    slider_changed = true;
                }
                slider_active |= ImGui::IsItemActive();
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Opacity of the back surface\nLower = more transparent from behind");
                }

                if (ImGui::SliderFloat("Back Darken", &quality_.back_darken, 0.5f, 1.0f, "%.2f")) {
                    slider_changed = true;
                }
                slider_active |= ImGui::IsItemActive();
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Darken back surface (simulates less light)\nLower = darker back");
                }

                if (ImGui::SliderFloat("Shell Edge", &quality_.shell_edge_threshold, 0.02f, 0.3f, "%.2f")) {
                    slider_changed = true;
                }
                slider_active |= ImGui::IsItemActive();
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Only add shell where gradient > threshold\nLower = more shell, Higher = only at sharp edges");
                }

                if (ImGui::Checkbox("Connect Walls", &quality_.connect_walls)) {
                    slider_changed = true;
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Add wall Gaussians at silhouettes\nConnects front to back at edges");
                }

                if (quality_.connect_walls) {
                    if (ImGui::SliderInt("Wall Segments", &quality_.wall_segments, 1, 6)) {
                        slider_changed = true;
                    }
                    slider_active |= ImGui::IsItemActive();
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip("Number of wall segments\nMore = smoother walls, more Gaussians");
                    }

                    if (ImGui::SliderFloat("Wall Opacity", &quality_.wall_opacity, 0.2f, 0.8f, "%.2f")) {
                        slider_changed = true;
                    }
                    slider_active |= ImGui::IsItemActive();
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip("Wall segment opacity\nHigher = more visible walls");
                    }
                }
            }

            // Adaptive Density - breaks stripe pattern at edges
            ImGui::Separator();
            ImGui::Text("Adaptive Density:");
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Adds extra Gaussians at edges with random offsets\nBreaks the stripe pattern visible from side views");
            }

            if (ImGui::Checkbox("Enable Density Fill", &quality_.adaptive_density)) {
                slider_changed = true;
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Add extra jittered Gaussians at depth edges\nFills gaps visible from oblique angles");
            }

            if (quality_.adaptive_density) {
                if (ImGui::SliderFloat("Density Threshold", &quality_.density_threshold, 0.02f, 0.2f, "%.2f")) {
                    slider_changed = true;
                }
                slider_active |= ImGui::IsItemActive();
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Gradient threshold for adding extra Gaussians\nLower = more areas get extra density");
                }

                if (ImGui::SliderInt("Extra Count", &quality_.density_extra, 1, 8)) {
                    slider_changed = true;
                }
                slider_active |= ImGui::IsItemActive();
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Number of extra Gaussians per edge point\nMore = better gap filling, more Gaussians");
                }

                if (ImGui::SliderFloat("Position Jitter", &quality_.density_jitter, 0.2f, 1.5f, "%.2f")) {
                    slider_changed = true;
                }
                slider_active |= ImGui::IsItemActive();
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Random position offset amount\nHigher = more spread, better stripe breaking");
                }
            }

            // Handle preview/full quality switching (AFTER all sliders)
            if (slider_changed) {
                if (slider_active) {
                    // While dragging: use fast preview
                    reprocess_image(true);
                    pending_reprocess_ = true;
                } else {
                    // Single click change: full quality
                    reprocess_image(false);
                    pending_reprocess_ = false;
                }
            }

            // When slider released, do full quality reprocess
            if (!slider_active && pending_reprocess_) {
                reprocess_image(false);
                pending_reprocess_ = false;
            }

            ImGui::Separator();

            if (ImGui::Button("Reset Quality", ImVec2(-1, 0))) {
                quality_.subsample = 1;
                quality_.gaussian_size = 0.008f;
                quality_.opacity = 0.9f;
                quality_.depth_scale = 2.5f;
                quality_.depth_exponent = 0.7f;
                quality_.auto_quality = true;
                quality_.max_gaussians = 500000;
                // Reset SAAG settings
                quality_.use_saag = true;
                quality_.aspect_ratio = 5.0f;
                quality_.edge_threshold = 0.15f;
                quality_.edge_shrink = 0.3f;
                quality_.gradient_scale = 50.0f;
                quality_.normal_strength = 1.0f;
                // Reset Silhouette Wrapping settings
                quality_.silhouette_wrap = true;
                quality_.wrap_layers = 3;
                quality_.wrap_spacing = 0.5f;
                quality_.wrap_opacity = 0.7f;
                quality_.wrap_edge_threshold = 0.15f;
                // Reset Volumetric Shell settings
                quality_.volumetric_shell = true;
                quality_.shell_thickness = 0.3f;
                quality_.back_opacity = 0.6f;
                quality_.back_darken = 0.8f;
                quality_.connect_walls = true;
                quality_.wall_segments = 3;
                quality_.wall_opacity = 0.5f;
                quality_.shell_edge_threshold = 0.1f;
                // Reset Adaptive Density settings
                quality_.adaptive_density = true;
                quality_.density_threshold = 0.08f;
                quality_.density_extra = 4;
                quality_.density_jitter = 0.6f;
                reprocess_image(false);
            }

            // Export section
            ImGui::Separator();
            ImGui::Text("Export:");
            ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
            ImGui::InputText("##exportpath", export_path_buffer_, sizeof(export_path_buffer_));
            if (ImGui::Button("Export HQ Image", ImVec2(-1, 0))) {
                if (export_path_buffer_[0] != '\0') {
                    export_image(export_path_buffer_);
                }
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Export current view at full quality\nSupports .ppm format (convert with ImageMagick)");
            }
        }

        ImGui::Separator();
        ImGui::Text("Test Clouds:");

        if (ImGui::Button("Load Test Cloud (1000)")) {
            load_test_cloud(1000, 3.0f);
        }

        if (ImGui::Button("Load Test Cloud (10000)")) {
            load_test_cloud(10000, 5.0f);
        }

        ImGui::Separator();
        ImGui::Text("Camera:");

        if (ImGui::Button("Reset Camera")) {
            camera_target_ = glm::vec3(0.0f);
            camera_distance_ = 5.0f;
            camera_yaw_ = 0.0f;
            camera_pitch_ = 0.3f;
            needs_rerender_ = true;
        }

        ImGui::Separator();
        ImGui::Text("Controls:");
        ImGui::BulletText("Left drag: Orbit");
        ImGui::BulletText("Right drag: Pan");
        ImGui::BulletText("Scroll: Zoom");
        ImGui::BulletText("R: Reset camera");
        ImGui::BulletText("Esc: Exit");
    }
    ImGui::End();
}

void Viewer::run() {
    auto last_time = std::chrono::high_resolution_clock::now();
    int frame_count = 0;
    float fps_timer = 0.0f;

    while (running_ && !glfwWindowShouldClose(window_)) {
        auto frame_start = std::chrono::high_resolution_clock::now();

        glfwPollEvents();
        process_input();

        // Render the Gaussian scene if needed
        render_frame();

        // Start ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // Render UI
        render_ui();

        // Finish ImGui frame
        ImGui::Render();

        // Clear and draw
        int display_w, display_h;
        glfwGetFramebufferSize(window_, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(0.05f, 0.05f, 0.07f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window_);

        // Calculate FPS
        auto frame_end = std::chrono::high_resolution_clock::now();
        stats_.frame_time_ms = std::chrono::duration<float, std::milli>(frame_end - frame_start).count();

        frame_count++;
        fps_timer += std::chrono::duration<float>(frame_end - last_time).count();
        last_time = frame_end;

        if (fps_timer >= 1.0f) {
            stats_.fps = static_cast<float>(frame_count) / fps_timer;
            frame_count = 0;
            fps_timer = 0.0f;
        }
    }
}

bool Viewer::is_running() const {
    return running_ && window_ && !glfwWindowShouldClose(window_);
}

void Viewer::close() {
    running_ = false;
}

} // namespace fresnel
