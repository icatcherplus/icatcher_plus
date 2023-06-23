import csv
from pathlib import Path
import pytest
import cv2
import numpy as np
from icatcher import version
from icatcher.face_detector import (
    process_frames,
    threshold_faces,
    extract_bboxes,
    find_bboxes,
)
from PIL import Image
import pooch
from batch_face import RetinaFace


@pytest.fixture
def retina_model():
    #download models that will be tested. these will be cached
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
    model = RetinaFace(
        gpu_id=-1, model_path=retina_model_file, network="resnet50"
    )
    return model


def test_process_frames():
    video_path = Path("tests", "test_data", "fd_video.mp4")
    test_cap = cv2.VideoCapture(str(video_path))
    ret, frame = test_cap.read()
    assert ret
    assert frame is not None  # testing that video is read in correctly

    test_frames = range(0, int(test_cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    h_start_at, w_start_at, w_end_at, h_end_at = 0, 0, 640, 480
    processed_frames = process_frames(
        test_cap, test_frames, h_start_at, h_end_at, w_start_at, w_end_at
    )

    assert len(processed_frames) == len(
        test_frames
    )  # testing that no frames were lost in process
    assert isinstance(
        processed_frames[0], np.ndarray
    )  # testing that all processed image frames are np arrays


@pytest.mark.parametrize(
    "filename, num_bounding_boxes",
    [
        ("no_face.jpg", 0),
        ("one_face_normal.jpg", 1),
        ("three_faces.jpg", 3),
    ],
)
def test_retina_face(filename, num_bounding_boxes, retina_model):
    face_detector_model = retina_model
    with Image.open(Path("tests", "test_data", filename)) as img:
        img_np = cv2.cvtColor(
            np.array(img), cv2.COLOR_RGB2BGR
        )  # changes image to mirror cv2 frame
    frame_height, frame_width = img_np.shape[0], img_np.shape[1]
    faces = face_detector_model(img_np)
    faces = [face for face in faces if face[-1] >= 0.9]
    bboxes = extract_bboxes(faces, frame_height, frame_width)
    assert (
        (len(bboxes) == num_bounding_boxes)
        if num_bounding_boxes > 0
        else bboxes is None
    )


# used for testing threshold faces
@pytest.fixture
def faces1():
    return [[(0, 0, 10, 10, 0.9), (10, 10, 20, 20, 0.8)], []]
@pytest.fixture
def faces2():
    return [[(0, 0, 10, 10, 0.9)], []]
@pytest.fixture
def all_faces():
    return [
        [(0, 0, 10, 10, 0.9), (10, 10, 20, 20, 0.8)],
        [(30, 30, 40, 40, 0.7), (40, 40, 50, 50, 0.6)],
    ]

@pytest.mark.parametrize(
    "confidence_threshold,output",
    [
        (0.75, 'faces1'),
        (0.9, 'faces2'),
        (0.5, 'all_faces'),
    ],
)
def test_threshold_faces(confidence_threshold, output, all_faces, request):
    assert threshold_faces(all_faces, confidence_threshold) == request.getfixturevalue(output)


def test_find_bboxes(retina_model):
    video_path = Path("tests", "test_data", "fd_video.mp4")
    test_cap = cv2.VideoCapture(str(video_path))
    test_frames = range(0, int(test_cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    h_start_at, w_start_at, w_end_at, h_end_at = 0, 0, 640, 480

    processed_frames = process_frames(
        test_cap, test_frames, h_start_at, h_end_at, w_start_at, w_end_at
    )
    frame_height, frame_width = (
        processed_frames[0].shape[0],
        processed_frames[0].shape[1],
    )
    face_detector_model = retina_model

    class OptContainer:
        def __init__(self):
            self.fd_batch_size = 16
            self.fd_confidence_threshold = 0.9

    test_opt = OptContainer()

    # num_cpus = mp.cpu_count()
    # num_frames_to_process = num_cpus * 16
    num_frames_to_process = 32
    # if num_frames_to_process > 173:  # this is the length of total video
    #     num_frames_to_process = 173
    processed_frames = processed_frames[: num_frames_to_process - 1]
    faces = find_bboxes(face_detector_model, test_opt, processed_frames)
    # faces = parallelize_face_detection(face_detector=face_detector_model, frames=processed_frames,
    # num_cpus=num_cpus, opt=test_opt)
    # faces = [item for sublist in faces for item in sublist]
    master_bboxes = [
        extract_bboxes(face_group, frame_height, frame_width) for face_group in faces
    ]

    # read in manual annotation
    with open(Path("tests", "test_data", "test_video_manual_annotation.csv"), 'r') as file:
        reader = csv.reader(file)
        next(reader)  # skip the first line
        second_line = next(reader)  # get the second line
        ground_truth = [int(x) for x in second_line]  # convert to list of integers

    ground_truth = ground_truth[: num_frames_to_process - 1]
    assert len(ground_truth) == len(master_bboxes)

    # get face counts of retina face w/ parallelization
    retina_face_counts = [
        len(face) if face is not None else 0 for face in master_bboxes
    ]
    matching_percentage = (
        sum(x == y for x, y in zip(ground_truth, retina_face_counts))
        / len(retina_face_counts)
    ) * 100
    assert (
        matching_percentage >= 95
    )  # make sure retina face has at least 95% accuracy on face counts


def test_extract_bboxes():
    face_group_1 = [
        (
            np.array([166.71333, 154.89905, 277.6311, 287.06418]),
            np.array(
                [
                    [180.20674, 221.42831],
                    [224.93073, 200.5159],
                    [197.3573, 222.8472],
                    [192.64197, 260.84647],
                    [222.6668, 244.67502],
                ]
            ),
            0.9996675,
        ),
        (
            np.array([80.74863, 179.17978, 158.89856, 305.16626]),
            np.array(
                [
                    [149.29008, 234.86317],
                    [152.41891, 233.67021],
                    [160.06853, 260.93158],
                    [146.51126, 284.67035],
                    [150.32614, 283.61874],
                ]
            ),
            0.96128803,
        ),
    ]
    frame_height, frame_width = 1000, 1000
    face_bboxes = extract_bboxes(face_group_1, frame_height, frame_width)
    assert np.array_equal(
        face_bboxes[0], np.array([166, 154, 110, 132])
    ) and np.array_equal(face_bboxes[1], np.array([80, 179, 78, 125]))

    # testing nothing in face_group
    face_group_2 = []
    assert extract_bboxes(face_group_2, frame_height, frame_width) is None
