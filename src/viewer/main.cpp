#include "viewer.hpp"
#include <iostream>
#include <cstring>
#include <cctype>

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
        // Detect file type by extension
        std::string ext;
        auto dot_pos = image_path.rfind('.');
        if (dot_pos != std::string::npos) {
            ext = image_path.substr(dot_pos);
            for (auto& c : ext) c = std::tolower(c);
        }

        bool success = false;
        if (ext == ".ply" || ext == ".bin") {
            // Load pre-computed Gaussian file
            std::cout << "Loading Gaussian file: " << image_path << "\n";
            success = viewer.load_gaussian_file(image_path);
        } else {
            // Load as image file
            std::cout << "Loading image: " << image_path << "\n";
            success = viewer.load_image(image_path);
        }

        if (!success) {
            std::cerr << "Failed to load file, falling back to test cloud\n";
            viewer.load_test_cloud(gaussian_count, 3.0f);
        }
    } else {
        // Load test cloud
        std::cout << "Usage: fresnel_viewer [file_path] [gaussian_count]\n";
        std::cout << "  file_path: Image file (JPG, PNG) or Gaussian file (.ply, .bin)\n";
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
