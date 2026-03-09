"utils function for snakemake scripts"

from typing import List, Dict, Union

def check_paths_for_warnings(paths: Union[List, Dict, str]) -> Union[List, Dict, str]:
    """
    Remove double slashes from paths in the input, which can be a string, list, or dictionary.

    Parameters:
    paths (Union[List, Dict, str]): The input paths which can be a single string, a list of strings, or a dictionary with string values.

    Returns:
    Union[List, Dict, str]: The input paths with double slashes removed.

    Raises:
    ValueError: If the input dictionary contains nested dictionaries.
    NotImplementedError: If the input is not a string, list, or dictionary.
    """
    # Check if the input is a string
    if isinstance(paths, str):
        # Replace double slashes with a single slash
        paths = paths.replace("//", "/")
    # Check if the input is a dictionary
    elif isinstance(paths, dict):
        for key, value in paths.items():
            # Raise an error if the dictionary contains nested dictionaries
            if isinstance(value, dict):
                raise ValueError("This function is not designed to handle nested dictionaries")
            # Replace double slashes in each string value of the dictionary
            paths[key] = [f"{item}".replace("//", "/") for item in value]
    # Check if the input is a list
    elif isinstance(paths, list):
        for index, item in enumerate(paths):
            # Raise an error if the list contains dictionaries
            if isinstance(item, dict):
                raise ValueError("This function is not designed to handle nested dictionaries")
            # Replace double slashes in each string item of the list
            paths[index] = f"{item}".replace("//", "/")
    else:
        # Raise an error if the input is not a string, list, or dictionary
        raise NotImplementedError("This function is only designed to handle strings, lists, and dictionaries")
    
    return paths

def make_list_input_ready(lst:list) -> str:
    return f"{lst}".replace(" ", "")
    