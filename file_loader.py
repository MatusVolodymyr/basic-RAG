"""
This module was used before the API was added. At the moment, all its functionality has been replaced by FastAPI and it is not used.
"""

import os


def load_text_from_file(file_path):
    """Loads text from a specified file"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()
    return text


def load_texts_from_directory(directory_path, file_extension=".txt"):
    """
    Loads text content from all files with a specific extension in a directory into a dictionary.

    This function scans the specified directory for files with the given extension, reads their contents,
    and returns a dictionary where keys are file names and values are their text content.

    Args:
        directory_path (str): The path to the directory containing text files.
        file_extension (str, optional): The file extension to filter by. Defaults to ".txt".

    Returns:
        dict[str, str]: A dictionary mapping file names to their text content.

    Raises:
        FileNotFoundError: If the specified directory does not exist.
        ValueError: If no files with the given extension are found.
        OSError: If there is an issue reading a file.
    """
    if not os.path.isdir(directory_path):
        raise FileNotFoundError(f"Directory '{directory_path}' does not exist.")

    documents = {}
    found_files = False

    for filename in os.listdir(directory_path):
        if filename.endswith(file_extension):
            found_files = True
            file_path = os.path.join(directory_path, filename)
            try:
                documents[filename] = load_text_from_file(file_path)
            except OSError as e:
                print(f"Error reading file {file_path}: {e}")

    if not found_files:
        raise ValueError(
            f"No files with extension '{file_extension}' found in '{directory_path}'."
        )

    return documents
