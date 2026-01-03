#pragma once

#include "../vulkan/context.hpp"
#include "pipeline.hpp"
#include <kompute/Kompute.hpp>
#include <memory>
#include <vector>

namespace fresnel {

/**
 * GPURadixSort - GPU-based radix sort for Gaussian depth sorting
 *
 * Implements a 4-pass radix sort (8 bits per pass) entirely on GPU,
 * eliminating the CPU-GPU roundtrip bottleneck in the rendering pipeline.
 *
 * Algorithm:
 *   1. Extract depth keys from 2D Gaussians buffer
 *   2. For each of 4 passes (bits 0-7, 8-15, 16-23, 24-31):
 *      a. Histogram: Count digit occurrences per workgroup
 *      b. Prefix Sum: Compute global write positions
 *      c. Scatter: Reorder keys and indices
 *   3. Return sorted indices
 *
 * Memory layout:
 *   - Ping-pong buffers for keys and indices
 *   - Per-workgroup histograms (256 bins each)
 *   - Global prefix sum buffer
 */
class GPURadixSort {
public:
    explicit GPURadixSort(VulkanContext& ctx);
    ~GPURadixSort() = default;

    // Non-copyable
    GPURadixSort(const GPURadixSort&) = delete;
    GPURadixSort& operator=(const GPURadixSort&) = delete;

    /**
     * Initialize sort resources for given maximum element count
     * @param max_elements Maximum number of Gaussians to support
     */
    void init(size_t max_elements);

    /**
     * Sort Gaussians by depth using GPU radix sort
     *
     * @param gaussians_2d Input buffer (12 floats per Gaussian)
     *        Layout: [mean.x, mean.y, cov_inv.a/b/c, depth, color.rgb, opacity, radius, visibility]
     * @param num_gaussians Number of Gaussians to sort
     * @return Tensor containing sorted indices (front-to-back by depth)
     */
    std::shared_ptr<kp::TensorT<uint32_t>> sort(
        std::shared_ptr<kp::TensorT<float>> gaussians_2d,
        size_t num_gaussians
    );

    /**
     * Check if GPU sort is initialized and ready
     */
    bool is_initialized() const { return initialized_; }

    /**
     * Get the maximum supported element count
     */
    size_t max_elements() const { return max_elements_; }

private:
    void compile_shaders();
    void create_buffers(size_t max_elements);

    // Sort pass operations
    void extract_keys(std::shared_ptr<kp::TensorT<float>> gaussians_2d, size_t count);
    void histogram_pass(uint32_t shift, size_t count);
    void prefix_sum_pass(size_t num_workgroups);
    void scatter_pass(uint32_t shift, size_t count);
    void swap_buffers();

    VulkanContext& ctx_;
    ComputePipeline pipeline_;

    // Compiled SPIR-V shaders
    std::vector<uint32_t> extract_shader_;
    std::vector<uint32_t> histogram_shader_;
    std::vector<uint32_t> prefix_sum_shader_;
    std::vector<uint32_t> scatter_shader_;

    // GPU buffers - ping-pong for keys and indices
    std::shared_ptr<kp::TensorT<uint32_t>> keys_[2];
    std::shared_ptr<kp::TensorT<uint32_t>> indices_[2];

    // Auxiliary buffers
    std::shared_ptr<kp::TensorT<uint32_t>> histograms_;      // Per-workgroup histograms
    std::shared_ptr<kp::TensorT<uint32_t>> global_offsets_;  // Prefix sum results
    std::shared_ptr<kp::TensorT<uint32_t>> params_;          // Sort parameters

    int current_buffer_ = 0;
    size_t max_elements_ = 0;
    size_t max_workgroups_ = 0;
    bool initialized_ = false;

    // Constants optimized for AMD RDNA 3
    static constexpr size_t WORKGROUP_SIZE = 256;
    static constexpr size_t ELEMENTS_PER_THREAD = 4;
    static constexpr size_t RADIX_BITS = 8;
    static constexpr size_t NUM_BINS = 256;   // 2^RADIX_BITS
    static constexpr size_t NUM_PASSES = 4;   // 32-bit / RADIX_BITS
};

} // namespace fresnel
