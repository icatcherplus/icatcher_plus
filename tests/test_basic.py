import pytest
import numpy as np
import icatcher
from icatcher.cli import predict_from_video
from pathlib import Path

def test_parse_illegal_transitions():
    """
    tests handling the option "illegal transitions".
    """
    bad_path1 = Path("tests/test_data/illegal_transitions_bad1.csv")
    bad_path2 = Path("tests/test_data/illegal_transitions_bad2.csv")
    bad_path3 = Path("tests/test_data/illegal_transitions_bad3.csv")
    good_path = Path("tests/test_data/illegal_transitions_good.csv")
    illegal, corrected = icatcher.parsers.parse_illegal_transitions_file(good_path)
    with pytest.raises(ValueError):
        _, _ = icatcher.parsers.parse_illegal_transitions_file(bad_path1)
    with pytest.raises(ValueError):
        _, _ = icatcher.parsers.parse_illegal_transitions_file(bad_path2)
    with pytest.raises(ValueError):
        _, _ = icatcher.parsers.parse_illegal_transitions_file(bad_path3)
    answers = [1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1]
    confidences = [1.0] * len(answers)
    answers, confidences = icatcher.cli.fix_illegal_transitions(
        3, answers, confidences, illegal, corrected
    )
    assert True


def test_process_video():
    """
    tests processing a video file.
    """
    arguments = "tests/test_data/test.mp4"
    opt = icatcher.options.parse_arguments(arguments)
    source = Path(opt.source)
    (
        cap,
        framerate,
        resolution,
        h_start_at,
        h_end_at,
        w_start_at,
        w_end_at,
    ) = icatcher.video.process_video(source, opt)
    assert True


def test_mask():
    """
    tests masking an image.
    """
    image = np.random.random((256, 512, 3))
    masked = icatcher.draw.mask_regions(image, 0, 128, 0, 256)
    assert masked[:128, 256:, :].all() == 0


@pytest.mark.parametrize(
    "args_string",
    [
        "tests/test_data/test.mp4 --model icatcher+_lookit.pth --fd_model opencv_dnn --output_annotation tests/test_data --overwrite",
        "tests/test_data/test.mp4 --model icatcher+_lookit.pth --fd_model opencv_dnn --output_annotation tests/test_data",
        "tests/test_data/test.mp4 --model icatcher+_lookit.pth --fd_model opencv_dnn --output_annotation tests/test_data --mirror_annotation --overwrite",
        "tests/test_data/test.mp4 --model icatcher+_lookit.pth --fd_model opencv_dnn --output_annotation tests/test_data --output_format compressed --overwrite",
        "tests/test_data/test.mp4 --model icatcher+_lookit.pth --fd_model opencv_dnn --output_annotation tests/test_data --mirror_annotation --output_format compressed --overwrite",
    ],
)
def test_predict_from_video(args_string):
    """
    runs entire prediction pipeline with several command line options.
    """
    args = icatcher.options.parse_arguments(args_string)
    if not args.overwrite:
        try:
            predict_from_video(args)
        except FileExistsError: # should be raised if overwrite is False and file exists, which is expected since this is the second test
            return
    else:
        predict_from_video(args)
    if args.output_annotation:
        if args.output_format == "compressed":
            output_file = Path("tests/test_data/test.npz")
            data = np.load(output_file)
            predicted_classes = data["arr_0"]
            confidences = data["arr_1"]
        else:
            output_file = Path("tests/test_data/test.txt")
            with open(output_file, "r") as f:
                data = f.readlines()
            predicted_classes = [x.split(",")[1].strip() for x in data]
            predicted_classes = np.array([icatcher.classes[x] for x in predicted_classes])
            confidences = np.array([float(x.split(",")[2].strip()) for x in data])
        assert len(predicted_classes) == len(confidences)
        # assert len(predicted_classes) == 194 # hard coded number of frames in test video
        if args.mirror_annotation:
            assert (predicted_classes == 2).all()
        else:
            assert (predicted_classes == 1).all()