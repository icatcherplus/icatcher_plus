import os
from pathlib import Path
import gdown


def download_from_gdrive(file_id: str, output_name: str):
    """
    Checks if the model weights have been downloaded and, if not, downloads from google drive into the
    correct directory.
    :param file_id: id associated with the download file
    :param output_name: the name this file should be saved under
    :return: None
    """
    # Check if the file already exists in the local directory
    if os.path.exists(os.path.join(os.path.join(str(Path(__file__).parents[1]), 'reproduce/models/'), output_name)):
        print(f"File with ID {file_id} already exists in the local directory.")
        return

    os.mkdir(os.path.join(str(Path(__file__).parents[1]), 'reproduce/models/'))
    download_directory = os.path.join(str(Path(__file__).parents[1]), 'reproduce/models/')

    # Download the file
    url = f"https://drive.google.com/uc?id={file_id}"
    output = os.path.join(download_directory, output_name)
    gdown.download(url, output, quiet=False)

    print(f"File with ID {file_id} has been downloaded to the local directory.")