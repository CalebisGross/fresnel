cmake_minimum_required(VERSION 3.20)
project(fresnel VERSION 0.1.0 LANGUAGES CXX)

# C++20 standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Build options
option(FRESNEL_BUILD_TESTS "Build tests" ON)
option(FRESNEL_BUILD_EXAMPLES "Build examples" ON)

# Find Vulkan
find_package(Vulkan REQUIRED)

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

# Core library
add_library(fresnel_core STATIC
    src/core/vulkan/context.cpp
    src/core/compute/pipeline.cpp
)

target_include_directories(fresnel_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
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

# Test executable - verify Vulkan + Kompute works
if(FRESNEL_BUILD_TESTS)
    add_executable(test_vulkan_compute tests/test_vulkan_compute.cpp)
    target_link_libraries(test_vulkan_compute PRIVATE fresnel_core)
endif()
