from pathlib import Path


def test_folder_structure():
    """
    tests that the directory structure is consistent with expected structure
    @return:
    """
    root_path = Path("./..")
    folders = ["datasets", "face_classifier", "models", "plots"]
    files = ["data.py", "logger.py", "models.py", "options.py", "parsers.py", "preprocess.py", "test.py",
             "train.py", "README.md", "visualize.py", "environment.yml", ".gitignore"]
    for folder in folders:
        folder_to_test = Path(root_path, folder)
        assert (folder_to_test.is_dir())
    for file in files:
        file_to_test = Path(root_path, file)
        assert (file_to_test.is_file())
