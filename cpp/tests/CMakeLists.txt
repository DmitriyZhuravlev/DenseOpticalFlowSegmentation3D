cmake_minimum_required(VERSION 3.12)
project(DenseOpticalFlowSegmentation3DTest)

# Find the OpenCV package
find_package(OpenCV REQUIRED)

# Include directories for OpenCV headers
include_directories(${OpenCV_INCLUDE_DIRS})
# Add an include directory for your project
include_directories(../inc)

# Include Google Test
include(GoogleTest)


link_directories(/home/dzhura/ComputerVision/DenseOpticalFlowSegmentation3D/cpp/build/)

# Add a common test executable for all tests in the "tests" folder
file(GLOB TEST_SOURCES "*.cpp")
add_executable(all_tests ${TEST_SOURCES})



#find_library(SEGMENTATION3D_LIB Segmentation3D)
#if(SEGMENTATION3D_LIB)
    #target_link_libraries(all_tests PRIVATE ${SEGMENTATION3D_LIB})
#else()
    #message(FATAL_ERROR "Segmentation3D library not found.")
#endif()

# Link against Google Test and your project's libraries
target_link_libraries(all_tests gtest gtest_main Segmentation3D ${OpenCV_LIBS} fmt spdlog)

# Register all tests with Google Test
gtest_discover_tests(all_tests)

# Optionally, you can create a test target that runs all tests
add_custom_target(run_all_tests)
add_dependencies(run_all_tests all_tests)
