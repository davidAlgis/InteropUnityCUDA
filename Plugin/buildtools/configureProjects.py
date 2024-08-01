import os
import subprocess
import argparse
import json
from toolsInterop import run_command


def buildConfigWindows(script_dir, project_dir, build_type, cuda_path):
    cmake_args = ['cmake', "-B", "build",
                  f"-DCMAKE_GENERATOR_TOOLSET=cuda={cuda_path}", "-D CMAKE_EXPORT_COMPILE_COMMANDS=ON"]
    print("Execute cmake with args : ", cmake_args)
    run_command(cmake_args, cwd=project_dir)


def buildConfigUnix(script_dir, project_dir, build_type):
    cmake_args = ['cmake', "-B", "build",
                  "-D CMAKE_EXPORT_COMPILE_COMMANDS=ON"]
    print("Execute cmake with args : ", cmake_args)
    run_command(cmake_args, cwd=project_dir)


def save_build_type(build_type):
    print("Save new build config in build_config.json...")
    # Define the path to the JSON file
    build_config_file = os.path.join(
        os.path.dirname(__file__), 'build_config.json')

    # Create a dictionary with the build type
    build_config = {'build_type': build_type}

    # Save the dictionary to the JSON file
    with open(build_config_file, 'w') as f:
        json.dump(build_config, f)


def main():
    parser = argparse.ArgumentParser(description='Build configuration script.')
    parser.add_argument("-b", "--build-type", choices=['Debug', 'Release'], default='Debug',
                        help='Select the build type (default: debug)')
    parser.add_argument("-c", "--cuda-path", default="C://Program Files//NVIDIA GPU Computing Toolkit//CUDA//v12.3",
                        help='Path to the CUDA toolkit (default: C://Program Files//NVIDIA GPU Computing Toolkit//CUDA//v12.3)')
    args = parser.parse_args()
    # Get the directory of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # we create build directory
    project_dir = os.path.join(script_dir, '..')
    build_dir = os.path.join(project_dir, 'build')

    if not os.path.exists(build_dir):
        os.makedirs(build_dir)

    # Save the build type to a JSON file
    save_build_type(args.build_type)

    if os.name == 'nt':  # Check if the system is Windows
        buildConfigWindows(script_dir, project_dir,
                           args.build_type, args.cuda_path)
    else:  # Assuming if it's not Windows, it's a Unix-like system
        buildConfigUnix(script_dir, project_dir, args.build_type)


if __name__ == '__main__':
    main()
