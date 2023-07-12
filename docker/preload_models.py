import sys
import pooch

def load_models(version):
    GOODBOY = pooch.create(
        path=pooch.os_cache("icatcher_plus"),
        base_url="https://osf.io/h7svp/download",
        version=version,
        version_dev="main",
        env="ICATCHER_DATA_DIR",
        registry={"zip_content.txt": None, "icatcher+_models.zip": None},
        urls={
            "zip_content.txt": "https://osf.io/v4w53/download",
            "icatcher+_models.zip": "https://osf.io/h7svp/download",
        },
    )
    
    file_paths = GOODBOY.fetch(
        "icatcher+_models.zip", processor=pooch.Unzip()
    )


def strip_version(version_output):
    version = version_output.split("==")[1]
    return version


def main():
    version_output = sys.argv[1]
    version = strip_version(version_output)
    load_models(version)


if __name__ == '__main__':
    main()
  
