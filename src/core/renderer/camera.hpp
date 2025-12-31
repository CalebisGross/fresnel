#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace fresnel {

/**
 * Camera - View and projection for rendering
 *
 * Handles transformation from world space to clip space.
 * Uses right-handed coordinate system with Y-up.
 */
class Camera {
public:
    Camera()
        : position_(0.0f, 0.0f, 5.0f)
        , target_(0.0f)
        , up_(0.0f, 1.0f, 0.0f)
        , fov_y_(45.0f)
        , aspect_(1.0f)
        , near_(0.1f)
        , far_(100.0f)
        , width_(800)
        , height_(600)
    {
        update_matrices();
    }

    // Position and orientation
    void set_position(glm::vec3 pos) { position_ = pos; update_matrices(); }
    void set_target(glm::vec3 target) { target_ = target; update_matrices(); }
    void set_up(glm::vec3 up) { up_ = up; update_matrices(); }

    void look_at(glm::vec3 pos, glm::vec3 target, glm::vec3 up = glm::vec3(0, 1, 0)) {
        position_ = pos;
        target_ = target;
        up_ = up;
        update_matrices();
    }

    // Projection parameters
    void set_fov(float fov_degrees) { fov_y_ = fov_degrees; update_matrices(); }
    void set_aspect(float aspect) { aspect_ = aspect; update_matrices(); }
    void set_clip(float near, float far) { near_ = near; far_ = far; update_matrices(); }
    void set_viewport(uint32_t width, uint32_t height) {
        width_ = width;
        height_ = height;
        aspect_ = static_cast<float>(width) / static_cast<float>(height);
        update_matrices();
    }

    // Getters
    glm::vec3 position() const { return position_; }
    glm::vec3 target() const { return target_; }
    glm::vec3 forward() const { return glm::normalize(target_ - position_); }
    glm::vec3 right() const { return glm::normalize(glm::cross(forward(), up_)); }
    glm::vec3 up() const { return up_; }

    float fov() const { return fov_y_; }
    float aspect() const { return aspect_; }
    float near_plane() const { return near_; }
    float far_plane() const { return far_; }
    uint32_t width() const { return width_; }
    uint32_t height() const { return height_; }

    // Matrices
    const glm::mat4& view() const { return view_; }
    const glm::mat4& projection() const { return projection_; }
    const glm::mat4& view_projection() const { return view_projection_; }

    /**
     * Focal length in pixels (for Gaussian projection)
     * fx = width / (2 * tan(fov_y / 2) * aspect)
     * fy = height / (2 * tan(fov_y / 2))
     */
    glm::vec2 focal() const {
        float tan_half_fov = std::tan(glm::radians(fov_y_) * 0.5f);
        float fy = static_cast<float>(height_) / (2.0f * tan_half_fov);
        float fx = fy; // Assuming square pixels
        return glm::vec2(fx, fy);
    }

    /**
     * Principal point (image center in pixels)
     */
    glm::vec2 principal_point() const {
        return glm::vec2(width_ * 0.5f, height_ * 0.5f);
    }

    /**
     * Project a 3D point to 2D screen coordinates
     * Returns (x, y) in pixels, origin at top-left
     */
    glm::vec2 project(glm::vec3 world_pos) const {
        glm::vec4 clip = view_projection_ * glm::vec4(world_pos, 1.0f);
        if (clip.w <= 0.0f) return glm::vec2(-1.0f); // Behind camera

        glm::vec3 ndc = glm::vec3(clip) / clip.w;
        // NDC is [-1, 1], convert to pixels
        float x = (ndc.x * 0.5f + 0.5f) * width_;
        float y = (1.0f - (ndc.y * 0.5f + 0.5f)) * height_; // Flip Y
        return glm::vec2(x, y);
    }

    /**
     * Get depth in view space for a world position
     */
    float view_depth(glm::vec3 world_pos) const {
        glm::vec4 view_pos = view_ * glm::vec4(world_pos, 1.0f);
        return -view_pos.z; // Negate because camera looks down -Z
    }

private:
    void update_matrices() {
        view_ = glm::lookAt(position_, target_, up_);
        projection_ = glm::perspective(glm::radians(fov_y_), aspect_, near_, far_);
        view_projection_ = projection_ * view_;
    }

    glm::vec3 position_;
    glm::vec3 target_;
    glm::vec3 up_;

    float fov_y_;     // Vertical FOV in degrees
    float aspect_;    // Width / Height
    float near_;
    float far_;

    uint32_t width_;
    uint32_t height_;

    glm::mat4 view_;
    glm::mat4 projection_;
    glm::mat4 view_projection_;
};

} // namespace fresnel
