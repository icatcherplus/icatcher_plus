from functools import partial
import cv2
import numpy as np
from pathos.pools import ProcessPool
from icatcher import draw
import logging

def threshold_faces(all_faces: list, confidence_threshold: float):
    """
    Selects all faces whose confidence score exceeds the defined confidence threshold
    :param all_faces: list of all faces and confidence scores present in all frames
    :param confidence_threshold: float threshold that a confidence score must exceed for face box to be used
    :return: all faces that exceeded the defined confidence threshold
    """
    for i, face_group in enumerate(all_faces):
        face_group = [face for face in face_group if face[-1] >= confidence_threshold]
        all_faces[i] = face_group
    return all_faces


def extract_bboxes(face_group_entry, frame_height, frame_width):
    """
    Extracts the bounding box from the face detector output
    :param face_group_entry: a group of faces detected from the face detector
    :param frame_height: amount of pixels in height of frame
    :param frame_width: amount of pixels in width of frame
    :return: the bounding boxes associated with each face in the face group
    """
    bboxes = []
    if face_group_entry:
        for face in face_group_entry:
            if isinstance(face[0], tuple):
                face = list(face[0])
            bbox = face[0]
            bbox[0] = max(bbox[0], 0)  # left side of box
            bbox[1] = max(bbox[1], 0)  # top side of box
            if (
                bbox[0] >= frame_width or bbox[1] >= frame_height
            ):  # if they are larger than image size, bbox is invalid
                continue
            bbox[2] = min(bbox[2], frame_width)  # right side of box
            bbox[3] = min(bbox[3], frame_height)  # bottom of box
            # change to width and height
            bbox[2] -= bbox[0]
            bbox[3] -= bbox[1]
            # take minimum of
            bboxes.append(bbox.astype(int))
    if not bboxes:
        bboxes = None
    return bboxes


def process_frames(cap, frames, h_start_at, h_end_at, w_start_at, w_end_at):
    """
    Takes in all desired frames of video and does some preprocessing and outputs images before face detection.
    :param cap: the video capture
    :param frames: list of numbers corresponding to frames
    :param h_start_at: optional crop coordinate
    :param h_end_at: optional crop coordinate
    :param w_start_at: optional crop coordinate
    :param w_end_at: optional crop coordinate
    :param v
    :return: list of images corresponding to video frames
    """
    processed_frames = []
    for frame in frames:
        logging.debug("buffering frames {}/{}".format(frame, len(frames)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ret, image = cap.read()
        if ret:
            image = draw.mask_regions(image, h_start_at, h_end_at, w_start_at, w_end_at)
            processed_frames.append(image)
        else:
            return processed_frames
    return processed_frames


def find_bboxes(face_detector, opt, processed_frames):
    """
    Uses batch inference to detect faces in frames thresholded at a certain confidence score.
    :param face_detector: face detector model
    :param opt: options
    :param processed_frames: input images fed into face detector
    :return: list of all faces and confidence scores present in all frames
    """
    all_faces = []
    batched_frames = [
        processed_frames[i : i + opt.fd_batch_size]
        for i in range(0, len(processed_frames), opt.fd_batch_size)
    ]
    for frame_group in batched_frames:
        faces = face_detector(frame_group)
        all_faces += faces

    # threshold amount of faces, confidence level based on user input
    thresholded_faces = threshold_faces(all_faces, opt.fd_confidence_threshold)
    return thresholded_faces


def detect_face_opencv_dnn(net, frame, conf_threshold):
    """
    Uses a pretrained face detection model to generate facial bounding boxes,
    with the format [x, y, width, height] where [x, y] is the lower left coord
    :param net:
    :param frame:
    :param conf_threshold:
    :return:
    """
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = max(int(detections[0, 0, i, 3] * frame_width), 0)  # left side of box
            y1 = max(int(detections[0, 0, i, 4] * frame_height), 0)  # top side of box
            if (x1 >= frame_width or y1 >= frame_height):  # if they are larger than image size, bbox is invalid
                continue
            x2 = min(int(detections[0, 0, i, 5] * frame_width), frame_width)  # either right side of box or frame width
            y2 = min(int(detections[0, 0, i, 6] * frame_height), frame_height)  # either the bottom side of box of frame height
            bboxes.append([x1, y1, x2 - x1, y2 - y1])  # (left, top, width, height)
    return bboxes


def parallelize_face_detection(frames, face_detector, num_cpus, opt):
    """
    Parallelizes face detection among the amount of cpus specified
    :param frames: list of images corresponding to video frames
    :param face_detector: model used for face detection
    :param num_cpus: number of cpus for parallelization
    :param opt: options
    :return: list of information regarding faces of each frame analyzed
    """

    # create a process pool with the number of cpus specified
    pool = ProcessPool(ncpus=num_cpus)

    # split the frames into even groups to distribute to each cpu
    frame_batches = np.array_split(frames, num_cpus)

    # create partial function and map it to the frame batches
    find_bboxes_func = partial(find_bboxes, face_detector, opt)
    faces = pool.map(find_bboxes_func, frame_batches)
    pool.close()
    pool.join()
    pool.clear()
    return faces
