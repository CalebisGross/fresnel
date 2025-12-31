cmake_minimum_required(VERSION 3.20)
project(fresnel VERSION 0.1.0 LANGUAGES CXX C)

# C++20 standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Build options
option(FRESNEL_BUILD_TESTS "Build tests" ON)
option(FRESNEL_BUILD_EXAMPLES "Build examples" ON)
option(FRESNEL_BUILD_VIEWER "Build ImGui viewer" ON)

# Find Vulkan
find_package(Vulkan REQUIRED)

# Find OpenGL (for ImGui backend)
find_package(OpenGL REQUIRED)

# Fetch Kompute
include(FetchContent)
FetchContent_Declare(
    kompute
    GIT_REPOSITORY https://github.com/KomputeProject/kompute.git
    GIT_TAG v0.9.0
)
set(KOMPUTE_OPT_USE_BUILT_IN_VULKAN_HEADER OFF CACHE BOOL "" FORCE)
set(KOMPUTE_OPT_DISABLE_VK_DEBUG_LAYERS ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(kompute)

# Fetch GLM
FetchContent_Declare(
    glm
    GIT_REPOSITORY https://github.com/g-truc/glm.git
    GIT_TAG 1.0.1
)
FetchContent_MakeAvailable(glm)

# Fetch GLFW
FetchContent_Declare(
    glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG 3.4
)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(glfw)

# Fetch Dear ImGui
FetchContent_Declare(
    imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG v1.91.6
)
FetchContent_MakeAvailable(imgui)

# Create ImGui library
add_library(imgui STATIC
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_demo.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
)
target_include_directories(imgui PUBLIC
    ${imgui_SOURCE_DIR}
    ${imgui_SOURCE_DIR}/backends
)
target_link_libraries(imgui PUBLIC glfw OpenGL::GL)

# Core library
add_library(fresnel_core STATIC
    src/core/vulkan/context.cpp
    src/core/compute/pipeline.cpp
    src/core/renderer/renderer.cpp
    src/core/image.cpp
    src/core/depth/estimator.cpp
    src/core/features/feature_extractor.cpp
    src/core/pointcloud.cpp
)

target_include_directories(fresnel_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
    ${Vulkan_INCLUDE_DIRS}
)

target_link_libraries(fresnel_core PUBLIC
    kompute
    glm::glm
    Vulkan::Vulkan
)

# Main executable (placeholder for now)
add_executable(fresnel src/main.cpp)
target_link_libraries(fresnel PRIVATE fresnel_core)

# Test executables
if(FRESNEL_BUILD_TESTS)
    add_executable(test_vulkan_compute tests/test_vulkan_compute.cpp)
    target_link_libraries(test_vulkan_compute PRIVATE fresnel_core)

    add_executable(test_gaussian_renderer tests/test_gaussian_renderer.cpp)
    target_link_libraries(test_gaussian_renderer PRIVATE fresnel_core)

    add_executable(test_depth_estimator tests/test_depth_estimator.cpp)
    target_link_libraries(test_depth_estimator PRIVATE fresnel_core)

    add_executable(test_pointcloud tests/test_pointcloud.cpp)
    target_link_libraries(test_pointcloud PRIVATE fresnel_core)

    add_executable(test_feature_extractor tests/test_feature_extractor.cpp)
    target_link_libraries(test_feature_extractor PRIVATE fresnel_core)
endif()

# Viewer application with ImGui
if(FRESNEL_BUILD_VIEWER)
    add_executable(fresnel_viewer
        src/viewer/main.cpp
        src/viewer/viewer.cpp
    )
    target_link_libraries(fresnel_viewer PRIVATE fresnel_core imgui)
endif()
