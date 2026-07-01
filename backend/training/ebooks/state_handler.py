import json
import os


def get_path(name):
    path = f"{os.path.dirname(__file__)}/states/{name}"

    return path


def check_state(name):
    """Check if state file is present or not. If not, create one."""
    if not os.path.isfile(get_path(name)):
        print(f'State file "{name}" not present, creating...')
        with open(get_path(name), "w"):
            print(f'State file "{name}" created successfully!')


def read_state(name):
    """Read data from json state file."""
    try:
        with open(get_path(name), "r") as file:
            return json.load(file)
    except json.JSONDecodeError:
        print(f'State file "{name}" is not a valid JSON file, returning empty dictionary.')
        return {}
    except FileNotFoundError:
        check_state(name)
        return read_state(name)


def write_state(name, data):
    """Write data to json state file."""
    file = open(get_path(name), "w")
    json.dump(data, file)
    file.close()
