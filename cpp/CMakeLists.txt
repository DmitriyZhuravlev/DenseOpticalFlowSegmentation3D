cmake_minimum_required(VERSION 3.12)
project(DenseOpticalFlowSegmentation3D)

# Add C++11 standard support
set(CMAKE_CXX_STANDARD 17)

# Enable debug symbols
set(CMAKE_BUILD_TYPE Debug)

# Add the directory containing FindOpenCV.cmake to the CMake module path
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/CMakeModules")

# Find the OpenCV package
find_package(OpenCV REQUIRED)

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})

# Add an include directory for your project
include_directories(inc)

# Source files in the src directory
file(GLOB SOURCES "src/*.cpp")

# Add your source files to the project
add_executable(DenseOpticalFlowSegmentation3D ${SOURCES})

# Link against OpenCV
target_link_libraries(DenseOpticalFlowSegmentation3D ${OpenCV_LIBS})

# Specify the command for running via Valgrind
set(VALGRIND_CMD valgrind --leak-check=full --track-origins=yes ./DenseOpticalFlowSegmentation3D)

# Create a custom target for running via Valgrind
add_custom_target(run_valgrind
    COMMAND ${VALGRIND_CMD}
    DEPENDS DenseOpticalFlowSegmentation3D
    WORKING_DIRECTORY ${CMAKE_PROJECT_DIR}
    COMMENT "Running via Valgrind..."
)
