import os

SUPPORTED_EXTENSIONS = (".py", ".md")


def discover_source_files(path):
    file_paths = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(SUPPORTED_EXTENSIONS):
                file_path = os.path.join(root, file)
                file_paths.append(file_path)

    return file_paths
