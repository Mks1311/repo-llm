import os

extensions = (".py", ".md")


def list_files(path):
    file_paths = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(extensions):
                file_path = os.path.join(root, file)

                # with open(file_path, "r", encoding="utf-8") as f:
                #     line_count = sum(1 for _ in f)

                # print(f"{file_path} - {line_count} lines")
                file_paths.append(file_path)

    return file_paths