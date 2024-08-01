import subprocess
import sys
import os
import json


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def run_command(*args, **kwargs):
    process = subprocess.Popen(
        *args, **kwargs, stdout=sys.stdout, stderr=sys.stderr)
    stdout, stderr = process.communicate()
    return process.returncode


def get_build_type():
    # Define the path to the JSON file
    build_config_file = os.path.join(
        os.path.dirname(__file__), 'build_config.json')

    # Check if the JSON file exists
    if not os.path.exists(build_config_file):
        print("Build configuration file not found. Execute buildConfiguration.py first.")
        return None

    # Load the JSON file
    with open(build_config_file, 'r') as f:
        build_config = json.load(f)

    # Get the build type from the JSON file
    build_type = build_config.get('build_type', None)

    # Check if the build type is valid
    if build_type not in ['Debug', 'Release']:
        print(f"Invalid build type in configuration file : {build_type}.")
        return None
    print(f"Retrieve build type in build_config.json: {build_type}.")
    return build_type
