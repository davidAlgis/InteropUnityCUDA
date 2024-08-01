cmake_minimum_required(VERSION 3.18)
project(PluginInteropUnitySolution)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable CUDA language support if needed
enable_language(CUDA)

if(UNIX AND NOT APPLE)
    add_definitions(-DUNITY_LINUX)
endif()

set(UNITY_INTEROP_PLUGIN_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../InteropUnityCUDA/Assets/Runtime/Plugin")

# Add subdirectories for each project
add_subdirectory(Utilities)
add_subdirectory(PluginInteropUnityCUDA)
add_subdirectory(SampleBasic)

