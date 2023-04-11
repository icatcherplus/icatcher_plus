import cv2
import numpy as np
import multiprocessing as mp
from pathlib import Path
from reproduce.face_detector import process_frames, parallelize_face_detection, threshold_faces, extract_bboxes
from reproduce import video
from face_detection import RetinaFace
from PIL import Image


def test_process_frames():
    video_path = Path("video_test", "test_video.mp4")
    test_cap = cv2.VideoCapture(str(video_path))
    _, meta_data = video.is_video_vfr(video_path, get_meta_data=True)

    # get raw width for process_frames function
    raw_width = meta_data["width"]
    raw_height = meta_data["height"]
    test_frames = range(0, int(test_cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    h_start_at, w_start_at, w_end_at = 0, 0, raw_width

    processed_frames = process_frames(test_cap, test_frames, h_start_at, w_start_at, w_end_at)

    assert len(processed_frames) == len(test_frames)  # testing that no frames were lost in process
    assert isinstance(processed_frames[0], np.ndarray)  # testing that all processed image frames are np arrays
    assert processed_frames[0].shape == (raw_height, raw_width, 3)  # test that size of image is same size if no crop


def test_retina_face():
    face_detector_model_file = Path(str(Path(__file__).parents[1]), "reproduce", "models", "Resnet50_Final.pth")
    network_name = "resnet50"
    face_detector_model = RetinaFace(gpu_id=-1, model_path=face_detector_model_file, network=network_name)

    # bring in some example frames where faces are detected and not detected
    image_list = []
    for filename in ['frames_test/no_face.jpg', 'frames_test/one_face_normal.jpg', 'frames_test/three_faces.jpg']:
        with Image.open(filename) as img:
            # change back to format that cv2 processes when reading in video
            img_np = np.array(img)
            image_list.append(cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))

    no_face, one_face, three_faces = image_list

    # testing image with no faces:
    faces = face_detector_model(no_face)
    faces = [face for face in faces if face[-1] >= 0.9]
    bboxes = extract_bboxes(faces)
    assert bboxes is None

    # testing image with one face:
    faces = face_detector_model(one_face)
    faces = [face for face in faces if face[-1] >= 0.9]
    bboxes = extract_bboxes(faces)
    assert len(bboxes) == 1

    # testing image with three faces:
    faces = face_detector_model(three_faces)
    faces = [face for face in faces if face[-1] >= 0.9]
    bboxes = extract_bboxes(faces)
    assert len(bboxes) == 3


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


def test_parallelize_face_detection():
    video_path = Path("video_test", "test_video.mp4")
    test_cap = cv2.VideoCapture(str(video_path))
    _, meta_data = video.is_video_vfr(video_path, get_meta_data=True)
    raw_width = meta_data["width"]
    test_frames = range(0, int(test_cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    h_start_at, w_start_at, w_end_at = 0, 0, raw_width

    processed_frames = process_frames(test_cap, test_frames, h_start_at, w_start_at, w_end_at)

    # test for Retina Face
    face_detector_model_file = Path(str(Path(__file__).parents[1]), "reproduce", "models", "Resnet50_Final.pth")
    network_name = "resnet50"
    face_detector_model = RetinaFace(gpu_id=-1, model_path=face_detector_model_file, network=network_name)

    class Opt_Container:
        def __init__(self):
            self.fd_batch_size = 16
            self.fd_confidence_threshold = 0.9

    test_opt = Opt_Container()

    # test with max available computers
    num_cpus = mp.cpu_count()
    faces = parallelize_face_detection(face_detector=face_detector_model, frames=processed_frames, num_cpus=num_cpus, opt=test_opt)
    assert faces is not None
    # insert comparison against ground truth manual annotation here


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
