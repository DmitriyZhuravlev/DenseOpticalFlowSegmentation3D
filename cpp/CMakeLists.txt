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
#file(GLOB SOURCES "src/*.cpp")
set(SOURCES
    src/segment.cpp
)

# List all the source files except segment.cpp
set(LIBRARY_SOURCES
    src/lifting_3d.cpp
    src/graph.cpp
    src/draw.cpp
    # Add more source files here as needed
)

add_library(Segmentation3D STATIC
    ${LIBRARY_SOURCES}
)

#target_include_directories(DenseOpticalFlowSegmentation3D
    #PUBLIC
    #${CMAKE_CURRENT_SOURCE_DIR}/inc
#)

# Add your source files to the project
add_executable(DenseOpticalFlowSegmentation3D ${SOURCES})

# Link against OpenCV
target_link_libraries(DenseOpticalFlowSegmentation3D Segmentation3D ${OpenCV_LIBS} fmt)

