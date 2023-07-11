import pytest
import torch
from icatcher.cli import load_models, predict_from_video
from icatcher.options import parse_arguments


@pytest.mark.parametrize(
    "args_string, model_class_name",
    [
        ("tests/test_data/test.mp4",
        "RegNet"
        ),
        (
            "tests/test_data/test.mp4 --model icatcher+_lookit_regnet.pth",
            "RegNet",
        ),
        (
            "tests/test_data/test.mp4 --model icatcher+_lookit.pth",
            "ResNet",
        ),
    ],
)
def test_load_models(args_string, model_class_name):
    """
    Checks that when load_model is called with the gaze model:
        1. left unspecified, RegNet trained on Lookit is called
        2. specified as "icatcher+_lookit_regnet.pth", RegNet trained on Lookit is called
        3. specified as "icatcher+_lookit.pth", ResNet trained on Lookit is called
    """

    args = parse_arguments(args_string)
    gaze_model, _, _, _ = load_models(args)
    assert gaze_model.encoder_img.__class__.__name__ == model_class_name


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Requires GPU for running, otherwise will timeout fail.",
)
@pytest.mark.parametrize(
    "args_string",
    [
        "tests/test_data/test.mp4 --model icatcher+_lookit_regnet.pth --gpu_id=0",
        "tests/test_data/test.mp4 --model icatcher+_lookit.pth --gpu_id=0",
    ],
)
def test_predict_from_video(args_string):
    """
    Ensures that the entire prediction pipeline is run to completion with both gaze models.
    """
    args = parse_arguments(args_string)
    predict_from_video(args)
