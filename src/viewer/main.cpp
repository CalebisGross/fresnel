#include "viewer.hpp"
#include <iostream>
#include <cstring>

int main(int argc, char* argv[]) {
    std::cout << "=== Fresnel Viewer ===\n\n";

    fresnel::Viewer viewer;

    fresnel::Viewer::Settings settings;
    settings.window_width = 1600;
    settings.window_height = 900;
    settings.render_width = 1280;
    settings.render_height = 720;
    settings.background_color = glm::vec3(0.08f, 0.08f, 0.1f);

    if (!viewer.init(settings)) {
        std::cerr << "Failed to initialize viewer\n";
        return 1;
    }

    // Check command line arguments
    std::string image_path;
    int gaussian_count = 1000;

    for (int i = 1; i < argc; i++) {
        // Check if argument looks like a file path (contains . or /)
        std::string arg = argv[i];
        if (arg.find('.') != std::string::npos || arg.find('/') != std::string::npos) {
            image_path = arg;
        } else {
            int count = std::atoi(argv[i]);
            if (count > 0) gaussian_count = count;
        }
    }

    if (!image_path.empty()) {
        // Load real image
        std::cout << "Loading image: " << image_path << "\n";
        if (!viewer.load_image(image_path)) {
            std::cerr << "Failed to load image, falling back to test cloud\n";
            viewer.load_test_cloud(gaussian_count, 3.0f);
        }
    } else {
        // Load test cloud
        std::cout << "Usage: fresnel_viewer [image_path] [gaussian_count]\n";
        std::cout << "  image_path: Path to image file (JPG, PNG, etc.)\n";
        std::cout << "  gaussian_count: Number of test Gaussians (default: 1000)\n\n";
        std::cout << "Loading test cloud with " << gaussian_count << " Gaussians...\n";
        viewer.load_test_cloud(gaussian_count, 3.0f);
    }

    std::cout << "\nStarting viewer...\n";
    std::cout << "  Mouse: Left=Orbit, Right=Pan, Scroll=Zoom\n";
    std::cout << "  Keys: R=Reset, Esc=Exit\n\n";

    viewer.run();

    std::cout << "Viewer closed.\n";
    return 0;
}
