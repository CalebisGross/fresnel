#include "viewer.hpp"
#include <iostream>

int main(int argc, char* argv[]) {
    std::cout << "=== Fresnel Viewer ===\n\n";

    fresnel::Viewer viewer;

    fresnel::Viewer::Settings settings;
    settings.window_width = 1280;
    settings.window_height = 720;
    settings.render_width = 800;
    settings.render_height = 600;
    settings.background_color = glm::vec3(0.1f, 0.1f, 0.15f);

    if (!viewer.init(settings)) {
        std::cerr << "Failed to initialize viewer\n";
        return 1;
    }

    // Load initial test cloud
    int gaussian_count = 1000;
    if (argc > 1) {
        gaussian_count = std::atoi(argv[1]);
        if (gaussian_count <= 0) gaussian_count = 1000;
    }

    std::cout << "Loading test cloud with " << gaussian_count << " Gaussians...\n";
    viewer.load_test_cloud(gaussian_count, 3.0f);

    std::cout << "Starting viewer...\n";
    std::cout << "  Mouse: Left=Orbit, Right=Pan, Scroll=Zoom\n";
    std::cout << "  Keys: R=Reset, Esc=Exit\n\n";

    viewer.run();

    std::cout << "Viewer closed.\n";
    return 0;
}
