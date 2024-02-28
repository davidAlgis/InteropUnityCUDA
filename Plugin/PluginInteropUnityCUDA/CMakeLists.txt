cmake_minimum_required(VERSION 3.18)
project(PluginInteropUnityCUDA)

# Find CUDA package
find_package(CUDAToolkit REQUIRED)


# Define the dynamic library target
file(GLOB_RECURSE  SOURCES "src/*.cpp")
file(GLOB  SOURCES_GL "thirdParty/gl3w/src/*.c")
file(GLOB_RECURSE  HEADER "include/**.h")
add_library(PluginInteropUnityCUDA SHARED ${SOURCES} ${HEADER} ${SOURCES_GL})


target_include_directories(PluginInteropUnityCUDA PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/Buffer
    ${CMAKE_CURRENT_SOURCE_DIR}/include/RenderAPI
    ${CMAKE_CURRENT_SOURCE_DIR}/include/Texture
    ${CMAKE_CURRENT_SOURCE_DIR}/thirdParty/gl3w/include
    ${CMAKE_CURRENT_SOURCE_DIR}/thirdParty/unity/include
)
target_link_libraries(PluginInteropUnityCUDA PUBLIC Utilities OpenGL32 cudart)

target_compile_definitions(PluginInteropUnityCUDA PRIVATE PLUGINGLINTEROPCUDA_SHARED_EXPORTS)

# Set properties for Debug/Release configurations
set_target_properties(PluginInteropUnityCUDA PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_NAME}/../../"
    LIBRARY_OUTPUT_DIRECTORY "${PROJECT_NAME}/../../"
    ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_NAME}/../../"
    DEBUG_OUTPUT_NAME "d_${PROJECT_NAME}"
    RELEASE_OUTPUT_NAME "${PROJECT_NAME}"
    MINSIZEREL_OUTPUT_NAME "${PROJECT_NAME}"
    RELWITHDEBINFO_OUTPUT_NAME "${PROJECT_NAME}"
)
