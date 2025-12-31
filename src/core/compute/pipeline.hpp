#pragma once

#include "../vulkan/context.hpp"
#include <kompute/Kompute.hpp>
#include <vector>
#include <span>
#include <functional>

namespace fresnel {

/**
 * ComputePipeline - High-level interface for GPU compute operations
 *
 * Provides a simplified interface for running compute shaders via Kompute.
 * Handles tensor creation, shader compilation, and execution.
 */
class ComputePipeline {
public:
    explicit ComputePipeline(VulkanContext& ctx);
    ~ComputePipeline() = default;

    /**
     * Create a GPU tensor from host data
     * @param data Host data to upload
     * @return Shared pointer to the tensor
     */
    template<typename T>
    std::shared_ptr<kp::TensorT<T>> create_tensor(std::span<const T> data) {
        std::vector<T> vec(data.begin(), data.end());
        return ctx_.manager().tensorT<T>(vec);
    }

    /**
     * Create an empty GPU tensor of given size
     */
    template<typename T>
    std::shared_ptr<kp::TensorT<T>> create_tensor(size_t count) {
        std::vector<T> vec(count, T{});
        return ctx_.manager().tensorT<T>(vec);
    }

    /**
     * Run a compute shader with given tensors
     * @param spirv_code SPIR-V shader bytecode
     * @param tensors Tensors to bind to the shader
     * @param workgroup_x X dimension of workgroup dispatch
     * @param workgroup_y Y dimension of workgroup dispatch (default 1)
     * @param workgroup_z Z dimension of workgroup dispatch (default 1)
     */
    void dispatch(
        const std::vector<uint32_t>& spirv_code,
        const std::vector<std::shared_ptr<kp::Tensor>>& tensors,
        uint32_t workgroup_x,
        uint32_t workgroup_y = 1,
        uint32_t workgroup_z = 1
    );

    /**
     * Synchronize tensors from GPU to host memory
     */
    void sync_to_host(const std::vector<std::shared_ptr<kp::Tensor>>& tensors);

    /**
     * Synchronize tensors from host to GPU memory
     */
    void sync_to_device(const std::vector<std::shared_ptr<kp::Tensor>>& tensors);

    /**
     * Get the underlying Vulkan context
     */
    VulkanContext& context() { return ctx_; }

private:
    VulkanContext& ctx_;
};

/**
 * Compile GLSL compute shader to SPIR-V
 * Requires glslangValidator or shaderc in PATH
 *
 * @param glsl_source GLSL shader source code
 * @return SPIR-V bytecode
 */
std::vector<uint32_t> compile_glsl(const std::string& glsl_source);

} // namespace fresnel
