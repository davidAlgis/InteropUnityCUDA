import os
import shutil
import json
import argparse
from toolsInterop import run_command


def generate_unity_package(target_folder, version):
    # Include version in the target folder
    target_folder = f"{target_folder}{version}"

    # Check if the target folder exists
    if os.path.exists(target_folder):
        # Remove the old target folder and create a new one
        shutil.rmtree(target_folder)
    os.makedirs(target_folder)

    # Define the content of package.json
    json_content = {
        "name": "com.studio-nyx.interop-unity-cuda",
        "version": version,
        "displayName": "Interop Unity CUDA",
        "unity": "2021.1",
        "description": "Demonstrate interoperability between Unity Engine and CUDA.",
        "license": "MIT",
        "keywords": ["GPU", "CUDA", "OpenGL", "DX11", "Native-Plugin", "interoperability"],
        "author": {
            "name": "David Algis",
            "email": "david.algis@tutamail.com",
            "url": "https://github.com/davidAlgis"
        },
        "dependencies": {
            "com.unity.mathematics": "1.2.6"
        }
    }

    # Create package.json in the target folder
    package_json_path = os.path.join(target_folder, 'package.json')
    with open(package_json_path, 'w') as json_file:
        json.dump(json_content, json_file, indent=4)

    print(f"package.json created in {target_folder}")

    # Copy and paste content from another folder to the target folder
    source_folder = os.path.join(os.path.dirname(
        __file__), '..', '..', 'InteropUnityCUDA', 'Assets')
    print(f"Copy {source_folder} to {target_folder}...")
    shutil.copytree(source_folder, target_folder, dirs_exist_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate a folder that can be used as a Unity package.')
    parser.add_argument('--target-folder', type=str, default=os.path.join(os.path.dirname(__file__), '..', '..', "com.studio-nyx.interop-unity-cuda."),
                        help='Location of the folder where the Unity package will be generated, it automatically adds the version to it.')
    parser.add_argument('--version', type=str, default="1.0.1",
                        help='Version number to be appended to the target folder.')

    args = parser.parse_args()

    # Get the directory of the current script
    script_dir = os.path.dirname(__file__)

    # Configure and compile the project in Debug mode
    run_command(['python', os.path.join(
        script_dir, 'configureProjects.py'), '--build-type', 'Debug'])
    run_command(['python', os.path.join(script_dir, 'compileProjects.py')])

    # Configure and compile the project in Release mode
    run_command(['python', os.path.join(
        script_dir, 'configureProjects.py'), '--build-type', 'Release'])
    run_command(['python', os.path.join(script_dir, 'compileProjects.py')])

    # Generate the Unity package
    generate_unity_package(args.target_folder, args.version)
