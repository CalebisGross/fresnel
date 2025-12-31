#include "viewer.hpp"

#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <iostream>
#include <chrono>
#include <cmath>

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
        ImGui::Text("GPU: %s", vk_context_->device_info().name.c_str());
    }
    ImGui::End();

    // Controls panel
    ImGui::SetNextWindowPos(
        ImVec2(static_cast<float>(settings_.render_width) + 40, 220),
        ImGuiCond_FirstUseEver
    );
    ImGui::SetNextWindowSize(ImVec2(250, 200), ImGuiCond_FirstUseEver);

    if (ImGui::Begin("Controls")) {
        ImGui::Text("Mouse Controls:");
        ImGui::BulletText("Left drag: Orbit");
        ImGui::BulletText("Right drag: Pan");
        ImGui::BulletText("Scroll: Zoom");
        ImGui::Separator();
        ImGui::Text("Keyboard:");
        ImGui::BulletText("R: Reset camera");
        ImGui::BulletText("Esc: Exit");
        ImGui::Separator();

        if (ImGui::Button("Reset Camera")) {
            camera_target_ = glm::vec3(0.0f);
            camera_distance_ = 5.0f;
            camera_yaw_ = 0.0f;
            camera_pitch_ = 0.3f;
            needs_rerender_ = true;
        }

        if (ImGui::Button("Load Test Cloud (1000)")) {
            load_test_cloud(1000, 3.0f);
        }

        if (ImGui::Button("Load Test Cloud (10000)")) {
            load_test_cloud(10000, 5.0f);
        }
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
