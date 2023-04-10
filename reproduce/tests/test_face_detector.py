import cv2
from pathlib import Path
from reproduce.face_detector import process_frames, find_bboxes, threshold_faces, extract_bboxes


# TODO: make a test video... maybe 100 frames? can read this in for process_frames and other tests

def test_process_frames():
    video_path = Path("test_video.mp4")
    test_cap = cv2.VideoCapture(str(video_path))
    test_frames = range(0, 100)
    h_start_at, w_start_at, w_end_at = 0, 0, -1  # change this to be the max

    processed_frames = process_frames(test_cap, test_frames, h_start_at, w_start_at, w_end_at)

    image = []
    assert len(processed_frames) == 100
    assert type(processed_frames[0]) == type(image)  # whatever type an img is here, test to find out

    # can test size of output images depending on crop


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
    face_group_1 = [[[10, 20, 30, 40], "fake", 0.9], [[90, 20, 300, 240], "fake", 0.95]]
    assert extract_bboxes(face_group_1) == [[10, 20, 20, 20], [90, 20, 210, 220]]

    # testing nothing in face_group
    face_group_2 = []
    assert extract_bboxes(face_group_2) is None

    # testing with one face in group, floats, and tuple is face[0]
    face_group_3 = [[([1.4, 9.3, 53.5, 93.9], "fake", 0.98)]]
    # TODO: check if int() rounds to closest whole number or floors it
    assert extract_bboxes(face_group_3) == [[1, 9, 52, 85]]
