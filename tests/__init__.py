import pooch
from pathlib import Path
from icatcher import version
from batch_face import RetinaFace

# download models that will be tested. these will be cached
GOODBOY = pooch.create(path=pooch.os_cache("icatcher_plus"),
                           base_url="https://osf.io/h7svp/download",
                           version=version,
                           version_dev="main",
                           env="ICATCHER_DATA_DIR",
                           registry={"zip_content.txt": None,
                                     "icatcher+_models.zip": None},
                           urls={"zip_content.txt":"https://osf.io/v4w53/download",
                                 "icatcher+_models.zip":"https://osf.io/h7svp/download"})

file_paths = GOODBOY.fetch("icatcher+_models.zip",
                           processor=pooch.Unzip(),
                           progressbar=True)

file_names = [Path(x).name for x in file_paths]

# load whatever models that need to be tested here
retina_model_file = file_paths[file_names.index("Resnet50_Final.pth")]
retina_model = RetinaFace(
    gpu_id=-1, model_path=retina_model_file, network="resnet50"
)
