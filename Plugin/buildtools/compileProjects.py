import os
import subprocess

from toolsInterop import run_command, str2bool, get_build_type


def compileProjects(project_dir, buildType):
    cmake_args = ['cmake', "--build", "build"]
    cmake_args.append("--config")
    cmake_args.append(f"{buildType}")
    print("Execute cmake with args : ", cmake_args)
    ret = run_command(cmake_args, cwd=project_dir)
    return ret


def is_executable(path):
    return os.path.isfile(path) and os.access(path, os.X_OK)


def main():
    # Get the directory of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.join(script_dir, '..')
    build_status = 0

    buildType = get_build_type()

    build_status = compileProjects(project_dir, buildType)
    if build_status == 0:  # build successful
        print("Build successful.")
    else:
        print("Build failed.")


if __name__ == '__main__':
    main()
