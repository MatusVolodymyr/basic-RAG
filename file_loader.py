import os

def load_text_from_file(file_path):
    """
    Loads text from a specified file.
    
    Args:
        file_path (str): The path to the text file.
    
    Returns:
        str: The content of the file as a string.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

def load_texts_from_directory(directory_path, file_extension=".txt"):
    """
    Loads text from all files with a specific extension in a directory.
    
    Args:
        directory_path (str): The path to the directory containing text files.
        file_extension (str): The file extension to filter by.
    
    Returns:
        Dict{"file_name": str}: A Dictionary of text contents from the files.
    """
    documents = {}
    for filename in os.listdir(directory_path):
        if filename.endswith(file_extension):
            file_path = os.path.join(directory_path, filename)
            documents[file_path]= load_text_from_file(file_path)
    return documents
