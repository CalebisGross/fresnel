#include "core/image.hpp"
#include "core/features/feature_extractor.hpp"
#include <iostream>
#include <cmath>

/**
 * Test suite for feature extraction (DINOv2)
 *
 * Verifies:
 * 1. FeatureMap creation
 * 2. DINOv2 extractor availability
 * 3. Feature extraction (if model available)
 */

bool test_feature_map_creation() {
    std::cout << "[TEST] FeatureMap creation... ";

    // Create a 37x37 feature map with 384 channels (DINOv2-Small output size)
    fresnel::FeatureMap features(37, 37, 384);

    if (features.width() != 37 || features.height() != 37 || features.channels() != 384) {
        std::cout << "FAIL (wrong dimensions)\n";
        return false;
    }

    if (features.size() != 37 * 37 * 384) {
        std::cout << "FAIL (wrong size: " << features.size() << ")\n";
        return false;
    }

    // Set some values
    features.set(18, 18, 0, 1.5f);
    features.set(18, 18, 383, -0.5f);

    if (std::abs(features.get(18, 18, 0) - 1.5f) > 1e-6f) {
        std::cout << "FAIL (value mismatch at channel 0)\n";
        return false;
    }

    if (std::abs(features.get(18, 18, 383) - (-0.5f)) > 1e-6f) {
        std::cout << "FAIL (value mismatch at channel 383)\n";
        return false;
    }

    std::cout << "PASS\n";
    return true;
}

bool test_feature_map_access() {
    std::cout << "[TEST] FeatureMap access patterns... ";

    fresnel::FeatureMap features(10, 10, 8);

    // Fill with pattern: value = x + y * 10 + c * 100
    for (uint32_t y = 0; y < 10; y++) {
        for (uint32_t x = 0; x < 10; x++) {
            for (uint32_t c = 0; c < 8; c++) {
                features.set(x, y, c, static_cast<float>(x + y * 10 + c * 100));
            }
        }
    }

    // Verify using at() pointer access
    const float* ptr = features.at(5, 3);
    for (uint32_t c = 0; c < 8; c++) {
        float expected = 5 + 3 * 10 + c * 100;
        if (std::abs(ptr[c] - expected) > 1e-6f) {
            std::cout << "FAIL (at() access mismatch at channel " << c << ")\n";
            return false;
        }
    }

    // Verify using feature_at() vector access
    auto vec = features.feature_at(7, 2);
    if (vec.size() != 8) {
        std::cout << "FAIL (feature_at() wrong size)\n";
        return false;
    }

    for (uint32_t c = 0; c < 8; c++) {
        float expected = 7 + 2 * 10 + c * 100;
        if (std::abs(vec[c] - expected) > 1e-6f) {
            std::cout << "FAIL (feature_at() mismatch at channel " << c << ")\n";
            return false;
        }
    }

    std::cout << "PASS\n";
    return true;
}

bool test_dinov2_availability() {
    std::cout << "[TEST] DINOv2 extractor availability... ";

    fresnel::DINOv2Extractor extractor;

    std::cout << (extractor.is_ready() ? "AVAILABLE" : "NOT AVAILABLE");
    std::cout << " (" << extractor.name() << ", " << extractor.feature_dim() << " dim)\n";

    // This test always passes - just reports availability
    return true;
}

bool test_dinov2_extraction() {
    std::cout << "[TEST] DINOv2 feature extraction... ";

    fresnel::DINOv2Extractor extractor;

    if (!extractor.is_ready()) {
        std::cout << "SKIP (model not available)\n";
        return true;  // Not a failure, just skip
    }

    // Create a test image (100x100 RGB)
    fresnel::Image img(100, 100, 3);

    // Fill with a simple pattern
    for (uint32_t y = 0; y < 100; y++) {
        for (uint32_t x = 0; x < 100; x++) {
            float r = static_cast<float>(x) / 100.0f;
            float g = static_cast<float>(y) / 100.0f;
            float b = 0.5f;
            img.set_rgb(x, y, r, g, b);
        }
    }

    // Extract features
    fresnel::FeatureMap features = extractor.extract(img);

    if (features.empty()) {
        std::cout << "FAIL (empty feature map)\n";
        return false;
    }

    // Check dimensions (should be 37x37x384 for DINOv2-Small with 518x518 input)
    std::cout << "output: " << features.width() << "x" << features.height()
              << "x" << features.channels() << " ";

    if (features.channels() != 384) {
        std::cout << "FAIL (wrong feature dim, expected 384)\n";
        return false;
    }

    // Check that features are non-zero (model actually ran)
    float sum = 0.0f;
    for (size_t i = 0; i < std::min(features.size(), size_t(1000)); i++) {
        sum += std::abs(features.data()[i]);
    }

    if (sum < 1e-6f) {
        std::cout << "FAIL (all zeros - model didn't run?)\n";
        return false;
    }

    std::cout << "PASS\n";
    return true;
}

bool test_factory() {
    std::cout << "[TEST] Feature extractor factory... ";

    auto extractor = fresnel::create_feature_extractor();

    if (extractor) {
        std::cout << "created: " << extractor->name();
        std::cout << " (ready: " << (extractor->is_ready() ? "yes" : "no") << ")\n";
    } else {
        std::cout << "nullptr (no extractor available)\n";
    }

    // This test always passes - just reports what the factory created
    return true;
}

int main() {
    std::cout << "=== Feature Extractor Tests ===\n\n";

    int passed = 0;
    int total = 0;

    auto run_test = [&](bool (*test)()) {
        total++;
        if (test()) passed++;
    };

    run_test(test_feature_map_creation);
    run_test(test_feature_map_access);
    run_test(test_dinov2_availability);
    run_test(test_dinov2_extraction);
    run_test(test_factory);

    std::cout << "\n=== Results: " << passed << "/" << total << " passed ===\n";

    return (passed == total) ? 0 : 1;
}
