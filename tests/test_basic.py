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
    arguments = "tests/test_data/test_short.mp4"
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
    "args_string, result_file",
    [
        # (
        #     "tests/test_data/test_long.mp4 --model icatcher+_lookit.pth --fd_model opencv_dnn --output_annotation tests/test_data --overwrite",
        #     "tests/test_data/test_long_result.txt",
        # ),
        (
            "tests/test_data/test_short.mp4 --model icatcher+_lookit.pth --fd_model opencv_dnn --output_annotation tests/test_data --overwrite",
            "tests/test_data/test_short_result.txt",
        ),
        (
            "tests/test_data/test_short.mp4 --model icatcher+_lookit.pth --fd_model opencv_dnn --output_annotation tests/test_data",
            "tests/test_data/test_short_result.txt",
        ),
        (
            "tests/test_data/test_short.mp4 --model icatcher+_lookit.pth --fd_model opencv_dnn --output_annotation tests/test_data --illegal_transitions_path tests/test_data/illegal_transitions_short.csv --overwrite",
            "tests/test_data/test_short_illegal_result.txt",
        ),
        (
            "tests/test_data/test_short.mp4 --model icatcher+_lookit.pth --fd_model opencv_dnn --output_annotation tests/test_data --mirror_annotation --overwrite",
            "tests/test_data/test_short_result.txt",
        ),
        (
            "tests/test_data/test_short.mp4 --model icatcher+_lookit.pth --fd_model opencv_dnn --output_annotation tests/test_data --output_format compressed --overwrite",
            "tests/test_data/test_short_result.txt",
        ),
        (
            "tests/test_data/test_short.mp4 --model icatcher+_lookit.pth --fd_model opencv_dnn --output_annotation tests/test_data --mirror_annotation --output_format compressed --overwrite",
            "tests/test_data/test_short_result.txt",
        ),
        (
            "tests/test_data/test_short.mp4 --model icatcher+_lookit.pth --fd_model opencv_dnn --output_annotation tests/test_data --output_format ui --overwrite",
            "tests/test_data/test_short_result.txt",
        ),
    ],
)
def test_predict_from_video(args_string, result_file):
    """
    runs entire prediction pipeline with several command line options.
    note this uses the original paper models which are faster but less accurate.
    tests for the newer models is out of scope for this test.
    """
    result_file = Path(result_file)
    with open(result_file, "r") as f:
        gt_data = f.readlines()
    gt_classes = [x.split(",")[1].strip() for x in gt_data]
    gt_classes = np.array([icatcher.classes[x] for x in gt_classes])
    gt_confidences = np.array([float(x.split(",")[2].strip()) for x in gt_data])
    args = icatcher.options.parse_arguments(args_string)
    if not args.overwrite:
        try:
            predict_from_video(args)
        except FileExistsError:
            # should be raised if overwrite is False and file exists, which is expected since this is not the first test
            return
    else:
        predict_from_video(args)
    if args.output_annotation:
        if args.output_format == "compressed":
            output_file = Path("tests/test_data/{}.npz".format(Path(args.source).stem))
            data = np.load(output_file)
            predicted_classes = data["arr_0"]
            confidences = data["arr_1"]
        elif args.output_format == "raw_output":
            output_file = Path("tests/test_data/{}.txt".format(Path(args.source).stem))
            with open(output_file, "r") as f:
                data = f.readlines()
            predicted_classes = [x.split(",")[1].strip() for x in data]
            predicted_classes = np.array(
                [icatcher.classes[x] for x in predicted_classes]
            )
            confidences = np.array([float(x.split(",")[2].strip()) for x in data])
        elif args.output_format == "ui":
            output_file = Path(
                "tests/test_data/{}/labels.txt".format(Path(args.source).stem)
            )
            with open(output_file, "r") as f:
                data = f.readlines()
            predicted_classes = [x.split(",")[1].strip() for x in data]
            predicted_classes = np.array(
                [icatcher.classes[x] for x in predicted_classes]
            )
            confidences = np.array([float(x.split(",")[2].strip()) for x in data])
        assert len(predicted_classes) == len(confidences)
        assert len(predicted_classes) == len(gt_classes)
        if args.mirror_annotation:
            modfied_predicted_classes = predicted_classes.copy()
            # 999 is just a dummy value
            modfied_predicted_classes[modfied_predicted_classes == 1] = 999
            modfied_predicted_classes[modfied_predicted_classes == 2] = 1
            modfied_predicted_classes[modfied_predicted_classes == 999] = 2
            assert (modfied_predicted_classes == gt_classes).all()
            np.isclose(gt_confidences, confidences, 0.01).all()
        else:
            assert (predicted_classes == gt_classes).all()
            np.isclose(gt_confidences, confidences, 0.01).all()
