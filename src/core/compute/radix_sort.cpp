#include "radix_sort.hpp"
#include <stdexcept>
#include <iostream>

namespace fresnel {

// Shader: Extract depth keys from 2D Gaussians and initialize indices
static const char* EXTRACT_KEYS_SHADER = R"(
#version 450

layout(local_size_x = 256) in;

// Input: 2D Gaussians (12 floats each)
// [0-1]: mean, [2-4]: cov_inv, [5]: depth, [6-8]: color, [9]: opacity, [10]: radius, [11]: visible
layout(binding = 0) buffer Gaussians2D {
    float g2d[];
} gaussians;

// Output: Keys (quantized depth as uint32)
layout(binding = 1) buffer Keys {
    uint keys[];
} output_keys;

// Output: Indices (identity initially)
layout(binding = 2) buffer Indices {
    uint indices[];
} output_indices;

// Parameters: [0] = count
layout(binding = 3) buffer Params {
    uint params[];
} p;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    uint count = p.params[0];
    if (idx >= count) return;

    uint base = idx * 12;
    float visible = gaussians.g2d[base + 11];
    float depth = gaussians.g2d[base + 5];

    // Quantize depth to uint32 for radix sort
    // Invisible Gaussians get max depth (pushed to back after sort)
    uint key;
    if (visible < 0.5) {
        key = 0xFFFFFFFFu;  // Max value for invisible
    } else {
        // Depth is typically in range [0.1, 1000]
        // Clamp and normalize to [0, 0xFFFFFFFE]
        float normalized = clamp(depth / 1000.0, 0.0, 1.0);
        key = uint(normalized * 4294967294.0);  // 0xFFFFFFFE max
    }

    output_keys.keys[idx] = key;
    output_indices.indices[idx] = idx;
}
)";

// Shader: Compute histogram of 8-bit digit for each workgroup
static const char* HISTOGRAM_SHADER = R"(
#version 450

layout(local_size_x = 256) in;

// Shared memory for local histogram (256 bins)
shared uint local_histogram[256];

// Input: Keys to histogram
layout(binding = 0) buffer Keys {
    uint keys[];
} input_keys;

// Output: Per-workgroup histograms (num_workgroups * 256)
layout(binding = 1) buffer Histograms {
    uint histograms[];
} output_hist;

// Parameters: [0] = count, [1] = shift, [2] = num_workgroups
layout(binding = 2) buffer Params {
    uint params[];
} p;

void main() {
    uint tid = gl_LocalInvocationID.x;
    uint wid = gl_WorkGroupID.x;
    uint count = p.params[0];
    uint shift = p.params[1];
    uint num_workgroups = p.params[2];

    // Initialize shared histogram
    local_histogram[tid] = 0;
    barrier();

    // Calculate element range for this workgroup
    uint elements_per_wg = (count + num_workgroups - 1) / num_workgroups;
    uint start = wid * elements_per_wg;
    uint end = min(start + elements_per_wg, count);

    // Each thread processes multiple elements
    for (uint i = start + tid; i < end; i += 256) {
        uint key = input_keys.keys[i];
        uint digit = (key >> shift) & 0xFFu;
        atomicAdd(local_histogram[digit], 1);
    }

    barrier();

    // Write local histogram to global memory
    output_hist.histograms[wid * 256 + tid] = local_histogram[tid];
}
)";

// Shader: Compute prefix sum across workgroup histograms
static const char* PREFIX_SUM_SHADER = R"(
#version 450

layout(local_size_x = 256) in;

// In/Out: Per-workgroup histograms - modified in place to prefix sums
layout(binding = 0) buffer Histograms {
    uint histograms[];
} hist;

// Output: Global digit totals (256 values)
layout(binding = 1) buffer GlobalOffsets {
    uint offsets[];
} global_off;

// Parameters: [0] = num_workgroups
layout(binding = 2) buffer Params {
    uint params[];
} p;

shared uint digit_sums[256];
shared uint digit_prefix[256];

void main() {
    uint tid = gl_LocalInvocationID.x;
    uint num_workgroups = p.params[0];

    // Each thread handles one digit (256 threads for 256 digits)
    // Sum all workgroup histograms for this digit
    uint sum = 0;
    for (uint wg = 0; wg < num_workgroups; wg++) {
        uint val = hist.histograms[wg * 256 + tid];
        hist.histograms[wg * 256 + tid] = sum;  // Store exclusive prefix for this workgroup
        sum += val;
    }

    // Store digit total in shared memory
    digit_sums[tid] = sum;
    barrier();

    // Compute prefix sum across all 256 digits (single workgroup handles all)
    // Use simple sequential scan (only 256 elements, fast enough)
    if (tid == 0) {
        uint running = 0;
        for (uint d = 0; d < 256; d++) {
            digit_prefix[d] = running;
            running += digit_sums[d];
        }
    }
    barrier();

    // Write global offset for this digit
    global_off.offsets[tid] = digit_prefix[tid];
}
)";

// Shader: Scatter keys and indices to sorted positions
static const char* SCATTER_SHADER = R"(
#version 450

layout(local_size_x = 256) in;

// Shared memory for local offsets
shared uint local_offsets[256];

// Input keys and indices
layout(binding = 0) buffer KeysIn {
    uint keys[];
} keys_in;

layout(binding = 1) buffer IndicesIn {
    uint indices[];
} indices_in;

// Output keys and indices (ping-pong buffers)
layout(binding = 2) buffer KeysOut {
    uint keys[];
} keys_out;

layout(binding = 3) buffer IndicesOut {
    uint indices[];
} indices_out;

// Histogram prefix sums
layout(binding = 4) buffer Histograms {
    uint histograms[];
} hist;

// Global digit offsets
layout(binding = 5) buffer GlobalOffsets {
    uint offsets[];
} global_off;

// Parameters: [0] = count, [1] = shift, [2] = num_workgroups
layout(binding = 6) buffer Params {
    uint params[];
} p;

void main() {
    uint tid = gl_LocalInvocationID.x;
    uint wid = gl_WorkGroupID.x;
    uint count = p.params[0];
    uint shift = p.params[1];
    uint num_workgroups = p.params[2];

    // Load this workgroup's histogram offsets + global offsets into shared memory
    local_offsets[tid] = hist.histograms[wid * 256 + tid] + global_off.offsets[tid];
    barrier();

    // Calculate element range for this workgroup
    uint elements_per_wg = (count + num_workgroups - 1) / num_workgroups;
    uint start = wid * elements_per_wg;
    uint end = min(start + elements_per_wg, count);

    // Scatter elements to their sorted positions
    for (uint i = start + tid; i < end; i += 256) {
        uint key = keys_in.keys[i];
        uint digit = (key >> shift) & 0xFFu;
        uint idx = indices_in.indices[i];

        // Atomically get output position and increment
        uint pos = atomicAdd(local_offsets[digit], 1);

        keys_out.keys[pos] = key;
        indices_out.indices[pos] = idx;
    }
}
)";

GPURadixSort::GPURadixSort(VulkanContext& ctx)
    : ctx_(ctx)
    , pipeline_(ctx)
{}

void GPURadixSort::init(size_t max_elements) {
    if (max_elements == 0) {
        throw std::invalid_argument("max_elements must be > 0");
    }

    max_elements_ = max_elements;

    // Calculate workgroup count
    size_t elements_per_wg = WORKGROUP_SIZE * ELEMENTS_PER_THREAD;
    max_workgroups_ = (max_elements + elements_per_wg - 1) / elements_per_wg;

    compile_shaders();
    create_buffers(max_elements);

    initialized_ = true;
}

void GPURadixSort::compile_shaders() {
    try {
        extract_shader_ = compile_glsl(EXTRACT_KEYS_SHADER);
        histogram_shader_ = compile_glsl(HISTOGRAM_SHADER);
        prefix_sum_shader_ = compile_glsl(PREFIX_SUM_SHADER);
        scatter_shader_ = compile_glsl(SCATTER_SHADER);
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Failed to compile radix sort shaders: ") + e.what());
    }
}

void GPURadixSort::create_buffers(size_t max_elements) {
    // Ping-pong buffers for keys and indices
    keys_[0] = pipeline_.create_tensor<uint32_t>(max_elements);
    keys_[1] = pipeline_.create_tensor<uint32_t>(max_elements);
    indices_[0] = pipeline_.create_tensor<uint32_t>(max_elements);
    indices_[1] = pipeline_.create_tensor<uint32_t>(max_elements);

    // Histogram buffer: num_workgroups * 256 bins
    histograms_ = pipeline_.create_tensor<uint32_t>(max_workgroups_ * NUM_BINS);

    // Global offset buffer: 256 values
    global_offsets_ = pipeline_.create_tensor<uint32_t>(NUM_BINS);

    // Parameters buffer: reused across passes
    params_ = pipeline_.create_tensor<uint32_t>(4);
}

std::shared_ptr<kp::TensorT<uint32_t>> GPURadixSort::sort(
    std::shared_ptr<kp::TensorT<float>> gaussians_2d,
    size_t num_gaussians
) {
    if (!initialized_) {
        throw std::runtime_error("GPURadixSort not initialized");
    }

    if (num_gaussians == 0) {
        return indices_[0];  // Return empty
    }

    if (num_gaussians > max_elements_) {
        throw std::runtime_error("num_gaussians exceeds max_elements");
    }

    current_buffer_ = 0;

    // Step 1: Extract keys from depth values and initialize indices
    extract_keys(gaussians_2d, num_gaussians);

    // Step 2: Four radix sort passes (8 bits each)
    for (uint32_t pass = 0; pass < NUM_PASSES; pass++) {
        uint32_t shift = pass * RADIX_BITS;

        histogram_pass(shift, num_gaussians);
        prefix_sum_pass((num_gaussians + WORKGROUP_SIZE * ELEMENTS_PER_THREAD - 1) /
                        (WORKGROUP_SIZE * ELEMENTS_PER_THREAD));
        scatter_pass(shift, num_gaussians);
        swap_buffers();
    }

    // Return sorted indices (after 4 swaps, result is in current_buffer_)
    return indices_[current_buffer_];
}

void GPURadixSort::extract_keys(std::shared_ptr<kp::TensorT<float>> gaussians_2d, size_t count) {
    // Set parameters
    std::vector<uint32_t> params = {static_cast<uint32_t>(count), 0, 0, 0};
    auto params_tensor = pipeline_.create_tensor<uint32_t>(std::span<const uint32_t>(params));
    pipeline_.sync_to_device({params_tensor});

    // Dispatch extract shader
    uint32_t num_groups = (count + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
    pipeline_.dispatch(
        extract_shader_,
        {gaussians_2d, keys_[0], indices_[0], params_tensor},
        num_groups
    );
}

void GPURadixSort::histogram_pass(uint32_t shift, size_t count) {
    size_t num_workgroups = (count + WORKGROUP_SIZE * ELEMENTS_PER_THREAD - 1) /
                            (WORKGROUP_SIZE * ELEMENTS_PER_THREAD);

    // Set parameters: [count, shift, num_workgroups]
    std::vector<uint32_t> params = {
        static_cast<uint32_t>(count),
        shift,
        static_cast<uint32_t>(num_workgroups),
        0
    };
    auto params_tensor = pipeline_.create_tensor<uint32_t>(std::span<const uint32_t>(params));
    pipeline_.sync_to_device({params_tensor});

    // Clear histograms
    std::vector<uint32_t> zeros(num_workgroups * NUM_BINS, 0);
    auto hist_zeros = pipeline_.create_tensor<uint32_t>(std::span<const uint32_t>(zeros));
    // Copy zeros to histogram buffer by recreating it
    histograms_ = pipeline_.create_tensor<uint32_t>(std::span<const uint32_t>(zeros));
    pipeline_.sync_to_device({histograms_});

    // Dispatch histogram shader
    pipeline_.dispatch(
        histogram_shader_,
        {keys_[current_buffer_], histograms_, params_tensor},
        num_workgroups
    );
}

void GPURadixSort::prefix_sum_pass(size_t num_workgroups) {
    // Set parameters: [num_workgroups]
    std::vector<uint32_t> params = {static_cast<uint32_t>(num_workgroups), 0, 0, 0};
    auto params_tensor = pipeline_.create_tensor<uint32_t>(std::span<const uint32_t>(params));
    pipeline_.sync_to_device({params_tensor});

    // Clear global offsets
    std::vector<uint32_t> zeros(NUM_BINS, 0);
    global_offsets_ = pipeline_.create_tensor<uint32_t>(std::span<const uint32_t>(zeros));
    pipeline_.sync_to_device({global_offsets_});

    // Dispatch prefix sum shader (single workgroup, 256 threads)
    pipeline_.dispatch(
        prefix_sum_shader_,
        {histograms_, global_offsets_, params_tensor},
        1
    );
}

void GPURadixSort::scatter_pass(uint32_t shift, size_t count) {
    size_t num_workgroups = (count + WORKGROUP_SIZE * ELEMENTS_PER_THREAD - 1) /
                            (WORKGROUP_SIZE * ELEMENTS_PER_THREAD);

    // Set parameters: [count, shift, num_workgroups]
    std::vector<uint32_t> params = {
        static_cast<uint32_t>(count),
        shift,
        static_cast<uint32_t>(num_workgroups),
        0
    };
    auto params_tensor = pipeline_.create_tensor<uint32_t>(std::span<const uint32_t>(params));
    pipeline_.sync_to_device({params_tensor});

    int next_buffer = 1 - current_buffer_;

    // Dispatch scatter shader
    pipeline_.dispatch(
        scatter_shader_,
        {keys_[current_buffer_], indices_[current_buffer_],
         keys_[next_buffer], indices_[next_buffer],
         histograms_, global_offsets_, params_tensor},
        num_workgroups
    );
}

void GPURadixSort::swap_buffers() {
    current_buffer_ = 1 - current_buffer_;
}

} // namespace fresnel
