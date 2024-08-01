import os
import subprocess
import argparse


def launch_unity_project(project_path, unity_bin):
    print(f"Launch Unity interop project at \"{project_path}\"...")
    command = [unity_bin, '-projectPath', project_path]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Launch a Unity project.')
    parser.add_argument("-u", "--unityBin", type=str, default="C://Program Files//Unity//Hub//Editor//2022.3.8f1//Editor//Unity.exe",
                        help='Path to the Unity executable.')
    args = parser.parse_args()

    project_path = os.path.join(os.path.dirname(
        __file__), '..', '..', 'InteropUnityCUDA')
    launch_unity_project(project_path, args.unityBin)
