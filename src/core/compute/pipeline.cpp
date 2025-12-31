#include "pipeline.hpp"
#include <stdexcept>
#include <fstream>
#include <cstdlib>
#include <array>
#include <filesystem>

namespace fresnel {

ComputePipeline::ComputePipeline(VulkanContext& ctx)
    : ctx_(ctx) {}

void ComputePipeline::dispatch(
    const std::vector<uint32_t>& spirv_code,
    const std::vector<std::shared_ptr<kp::Tensor>>& tensors,
    uint32_t workgroup_x,
    uint32_t workgroup_y,
    uint32_t workgroup_z
) {
    // Create algorithm with the shader and tensors
    auto algorithm = ctx_.manager().algorithm(
        tensors,
        spirv_code,
        kp::Workgroup({workgroup_x, workgroup_y, workgroup_z})
    );

    // Create and run sequence
    ctx_.manager().sequence()
        ->record<kp::OpAlgoDispatch>(algorithm)
        ->eval();
}

void ComputePipeline::sync_to_host(const std::vector<std::shared_ptr<kp::Tensor>>& tensors) {
    ctx_.manager().sequence()
        ->record<kp::OpTensorSyncLocal>(tensors)
        ->eval();
}

void ComputePipeline::sync_to_device(const std::vector<std::shared_ptr<kp::Tensor>>& tensors) {
    ctx_.manager().sequence()
        ->record<kp::OpTensorSyncDevice>(tensors)
        ->eval();
}

std::vector<uint32_t> compile_glsl(const std::string& glsl_source) {
    // Create temporary files for compilation
    auto temp_dir = std::filesystem::temp_directory_path();
    auto glsl_path = temp_dir / "fresnel_shader.comp";
    auto spirv_path = temp_dir / "fresnel_shader.spv";

    // Write GLSL source to temp file
    {
        std::ofstream out(glsl_path);
        if (!out) {
            throw std::runtime_error("Failed to create temp shader file");
        }
        out << glsl_source;
    }

    // Try glslangValidator first, then glslc (shaderc)
    std::string cmd;
    bool has_glslang = (std::system("which glslangValidator > /dev/null 2>&1") == 0);
    bool has_shaderc = (std::system("which glslc > /dev/null 2>&1") == 0);

    if (has_glslang) {
        cmd = "glslangValidator -V -o " + spirv_path.string() + " " + glsl_path.string() + " 2>&1";
    } else if (has_shaderc) {
        cmd = "glslc -o " + spirv_path.string() + " " + glsl_path.string() + " 2>&1";
    } else {
        throw std::runtime_error("No GLSL compiler found. Install glslang or shaderc.");
    }

    // Run compiler
    std::array<char, 256> buffer;
    std::string result;
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) {
        throw std::runtime_error("Failed to run GLSL compiler");
    }
    while (fgets(buffer.data(), buffer.size(), pipe) != nullptr) {
        result += buffer.data();
    }
    int ret = pclose(pipe);

    if (ret != 0) {
        throw std::runtime_error("GLSL compilation failed:\n" + result);
    }

    // Read SPIR-V binary
    std::ifstream spirv_file(spirv_path, std::ios::binary | std::ios::ate);
    if (!spirv_file) {
        throw std::runtime_error("Failed to read compiled SPIR-V");
    }

    auto size = spirv_file.tellg();
    spirv_file.seekg(0, std::ios::beg);

    std::vector<uint32_t> spirv(size / sizeof(uint32_t));
    spirv_file.read(reinterpret_cast<char*>(spirv.data()), size);

    // Cleanup temp files
    std::filesystem::remove(glsl_path);
    std::filesystem::remove(spirv_path);

    return spirv;
}

} // namespace fresnel
