import pytest
import numpy as np
import icatcher
from pathlib import Path

def test_parse_illegal_transitions():
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
    confidences = [1.0]*len(answers)
    answers, confidences = icatcher.cli.fix_illegal_transitions(3, answers, confidences, illegal, corrected)
    assert True

def test_process_video():
    arguments = "tests/test_data/test.mp4"
    opt = icatcher.options.parse_arguments(arguments)
    source = Path(opt.source)
    cap, framerate, resolution, \
    h_start_at, h_end_at, w_start_at, w_end_at = icatcher.video.process_video(source, opt)
    assert True

def test_mask():
    image = np.random.random((256, 512, 3))
    masked = icatcher.draw.mask_regions(image, 0, 128, 0, 256)
    assert masked[:128, 256:, :].all() == 0