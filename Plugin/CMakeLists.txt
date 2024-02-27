cmake_minimum_required(VERSION 3.18)
project(PluginInteropUnitySolution)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable CUDA language support if needed
enable_language(CUDA)

# Add subdirectories for each project
add_subdirectory(Utilities)
add_subdirectory(PluginInteropUnityCUDA)
add_subdirectory(SampleBasic)
