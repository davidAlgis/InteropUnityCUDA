cmake_minimum_required(VERSION 3.18)
project(Utilities)

# Find CUDA package
find_package(CUDAToolkit REQUIRED)


# Define the dynamic library target
file(GLOB SOURCES "src/*.cpp")
file(GLOB HEADER "include/**.h")
add_library(Utilities SHARED ${SOURCES})

target_link_directories(Utilities PUBLIC 
    ${CUDAToolkit_LIBRARY_DIR})


target_link_libraries(Utilities PUBLIC 
    CUDA::cudart)

target_include_directories(Utilities PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)


target_compile_definitions(Utilities PRIVATE UTILITIES_SHARED UTILITIES_SHARED_EXPORTS)

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

add_custom_command(TARGET Utilities POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    $<TARGET_FILE:Utilities>
    ${UNITY_INTEROP_PLUGIN_PATH}/$<CONFIG>/$<TARGET_FILE_NAME:Utilities>
)