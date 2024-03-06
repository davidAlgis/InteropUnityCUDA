cmake_minimum_required(VERSION 3.18)
project(SampleBasic)

# Find CUDA package
find_package(CUDAToolkit REQUIRED)


# Define the executable target
file(GLOB SOURCES "src/*.cpp" "src/*.cu")
file(GLOB HEADER "include/*.h")
add_library(SampleBasic SHARED ${SOURCES} ${HEADER})

target_include_directories(SampleBasic PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

# Linking with release libraries
target_link_libraries(SampleBasic optimized 
    Utilities
    OpenGL32
    PluginInteropUnityCUDA
    cudart
    cuda
)


if(UNIX AND NOT APPLE)
    add_definitions(-DUNITY_LINUX)
endif()


# Set properties for Debug/Release configurations
set_target_properties(SampleBasic PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_NAME}/../../"
    LIBRARY_OUTPUT_DIRECTORY "${PROJECT_NAME}/../../"
    ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_NAME}/../../"
    DEBUG_OUTPUT_NAME "d_${PROJECT_NAME}"
    RELEASE_OUTPUT_NAME "${PROJECT_NAME}"
    MINSIZEREL_OUTPUT_NAME "${PROJECT_NAME}"
    RELWITHDEBINFO_OUTPUT_NAME "${PROJECT_NAME}"
)