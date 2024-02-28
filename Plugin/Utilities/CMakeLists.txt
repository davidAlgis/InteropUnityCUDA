cmake_minimum_required(VERSION 3.18)
project(Utilities)

# Find CUDA package
find_package(CUDAToolkit REQUIRED)


# Define the dynamic library target
file(GLOB SOURCES "src/*.cpp")
file(GLOB HEADER "include/**.h")
add_library(Utilities SHARED ${SOURCES})

target_link_libraries(Utilities PRIVATE cudart)

target_include_directories(Utilities PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)


target_compile_definitions(Utilities PRIVATE UTILITIES_SHARED UTILITIES_SHARED_EXPORTS)

link_directories("${PROJECT_NAME}/../../")

# Set properties for Debug/Release configurations
set_target_properties(Utilities PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_NAME}/../../"
    ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_NAME}/../../"
    LIBRARY_OUTPUT_DIRECTORY "${PROJECT_NAME}/../../"
    DEBUG_OUTPUT_NAME "d_${PROJECT_NAME}"
    RELEASE_OUTPUT_NAME "${PROJECT_NAME}"
    MINSIZEREL_OUTPUT_NAME "${PROJECT_NAME}"
    RELWITHDEBINFO_OUTPUT_NAME "${PROJECT_NAME}"
)