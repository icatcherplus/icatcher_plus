import pytest
import os
import cv2
import numpy as np
import multiprocessing as mp
import pandas as pd
from pathlib import Path
from reproduce.face_detector import process_frames, parallelize_face_detection, threshold_faces, extract_bboxes, create_retina_model
from reproduce import video
from PIL import Image

# def make_opt(**kwargs):  # dict(framerate=20, path=..., ...) -> opt
#     return OptContainer ...

# used for testing threshold faces
all_faces = [
    [(0, 0, 10, 10, 0.9), (10, 10, 20, 20, 0.8)],
    [(30, 30, 40, 40, 0.7), (40, 40, 50, 50, 0.6)]
]

def test_process_frames():
    video_path = os.path.join(str(Path(__file__).parents[1]), "tests", "video_test", "test_video.mp4")
    test_cap = cv2.VideoCapture(str(video_path))

    # testing that video is read in correctly
    ret, frame = test_cap.read()
    assert ret
    assert frame is not None

    test_frames = range(0, int(test_cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    h_start_at, w_start_at, w_end_at = 0, 0, 640

    processed_frames = process_frames(test_cap, test_frames, h_start_at, w_start_at, w_end_at)

    assert len(processed_frames) == len(test_frames)  # testing that no frames were lost in process
    assert isinstance(processed_frames[0], np.ndarray)  # testing that all processed image frames are np arrays


@pytest.mark.parametrize('filename, num_bounding_boxes', [
    ('no_face.jpg', 0),
    ('one_face_normal.jpg', 1),
    ('three_faces.jpg', 3),
])
def test_retina_face(filename, num_bounding_boxes):
    face_detector_model = create_retina_model()
    os.path.join(str(Path(__file__).parents[1]), 'reproduce/models/')
    with Image.open(os.path.join(str(Path(__file__).parents[1]), 'tests', 'frames_test', filename)) as img:  # change image to mirror cv2 frame
        img_np = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    faces = face_detector_model(img_np)
    faces = [face for face in faces if face[-1] >= 0.9]
    bboxes = extract_bboxes(faces)
    assert (len(bboxes) == num_bounding_boxes) if num_bounding_boxes > 0 else bboxes is None


@pytest.mark.parametrize('confidence_threshold,output', [
    (0.75, [[(0, 0, 10, 10, 0.9), (10, 10, 20, 20, 0.8)], []]),
    (0.9, [[(0, 0, 10, 10, 0.9)], []]),
    (0.5, all_faces),
])
def test_threshold_faces(confidence_threshold, output):
    assert threshold_faces(all_faces, confidence_threshold) == output


def test_parallelize_face_detection():
    video_path = os.path.join(str(Path(__file__).parents[1]), "tests", "video_test", "test_video.mp4")
    test_cap = cv2.VideoCapture(str(video_path))
    test_frames = range(0, int(test_cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    h_start_at, w_start_at, w_end_at = 0, 0, 640

    processed_frames = process_frames(test_cap, test_frames, h_start_at, w_start_at, w_end_at)
    face_detector_model = create_retina_model()

    class Opt_Container:
        def __init__(self):
            self.fd_batch_size = 16
            self.fd_confidence_threshold = 0.9

    test_opt = Opt_Container()

    # test with max available computers
    num_cpus = mp.cpu_count()
    faces = parallelize_face_detection(face_detector=face_detector_model, frames=processed_frames, num_cpus=num_cpus, opt=test_opt)
    faces = [item for sublist in faces for item in sublist]
    master_bboxes = [extract_bboxes(face_group) for face_group in faces]

    # read in manual annotation
    os.path.join(str(Path(__file__).parents[1]), "tests", "video_test", "test_video_manual_annotation.csv")
    ground_truth = pd.read_csv(os.path.join(str(Path(__file__).parents[1]), "tests", "video_test", "test_video_manual_annotation.csv"))
    ground_truth = ground_truth.loc[0, :].values.flatten().tolist()
    assert len(ground_truth) == len(master_bboxes)

    # get face counts of retina face w/ parallelization
    retina_face_counts = [len(face) if face is not None else 0 for face in master_bboxes]

    matching_percentage = (sum(x == y for x, y in zip(ground_truth, retina_face_counts)) / len(retina_face_counts)) * 100
    assert matching_percentage >= 95  # make sure retina face has at least 95% accuracy on face counts


def test_extract_bboxes():
    face_group_1 = [(np.array([166.71333, 154.89905, 277.6311 , 287.06418]), np.array([[180.20674, 221.42831],
       [224.93073, 200.5159 ],
       [197.3573 , 222.8472 ],
       [192.64197, 260.84647],
       [222.6668 , 244.67502]]), 0.9996675), (np.array([ 80.74863, 179.17978, 158.89856, 305.16626]), np.array([[149.29008, 234.86317],
       [152.41891, 233.67021],
       [160.06853, 260.93158],
       [146.51126, 284.67035],
       [150.32614, 283.61874]]), 0.96128803)]
    face_bboxes = extract_bboxes(face_group_1)
    assert np.array_equal(face_bboxes[0], np.array([166, 154, 110, 132])) and np.array_equal(face_bboxes[1], np.array([80, 179, 78, 125]))

    # testing nothing in face_group
    face_group_2 = []
    assert extract_bboxes(face_group_2) is None
