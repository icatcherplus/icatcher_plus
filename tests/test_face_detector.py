import cv2
import numpy as np
from pathlib import Path
from reproduce.face_detector import process_frames, find_bboxes, threshold_faces, extract_bboxes


# TODO: make a test video... maybe 100 frames? can read this in for process_frames and other tests

# def test_process_frames():
#     video_path = Path("test_video.mp4")
#     test_cap = cv2.VideoCapture(str(video_path))
#     test_frames = range(0, 100)
#     h_start_at, w_start_at, w_end_at = 0, 0, -1  # change this to be the max
#
#     processed_frames = process_frames(test_cap, test_frames, h_start_at, w_start_at, w_end_at)
#
#     image = []
#     assert len(processed_frames) == 100
#     assert type(processed_frames[0]) == type(image)  # whatever type an img is here, test to find out
#
#     # can test size of output images depending on crop


def test_threshold_faces():
    all_faces = [
        [(0, 0, 10, 10, 0.9), (10, 10, 20, 20, 0.8)],
        [(30, 30, 40, 40, 0.7), (40, 40, 50, 50, 0.6)]
    ]
    confidence_threshold = 0.75
    expected_output = [
        [(0, 0, 10, 10, 0.9), (10, 10, 20, 20, 0.8)],
        []
    ]
    assert threshold_faces(all_faces, confidence_threshold) == expected_output
    assert threshold_faces(all_faces, 0.9) == [[(0, 0, 10, 10, 0.9)], []]
    assert threshold_faces(all_faces, 0.5) == all_faces


def test_find_bboxes():
    pass


def test_parallelize_face_detection():
    pass


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
