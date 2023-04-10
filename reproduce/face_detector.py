import cv2


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


def extract_bboxes(face_group_entry):
    """
    Extracts the bounding box from the face detector output
    :param face_group_entry: a group of faces detected from the face detector
    :return: the bounding boxes associated with each face in the face group
    """
    bboxes = []
    if face_group_entry:
        for face in face_group_entry:
            if type(face[0]) is tuple:
                face = list(face[0])
            bbox = face[0]
            # change to width and height
            bbox[2] -= bbox[0]
            bbox[3] -= bbox[1]
            bboxes.append(bbox.astype(int))
    if not bboxes:
        bboxes = None
    return bboxes


def process_frames(cap, frames, h_start_at, w_start_at, w_end_at):
    """
    Takes in all desired frames of video and does some preprocessing and outputs images before face detection.
    :param cap: the video capture
    :param frames: list of numbers corresponding to frames
    :param h_start_at: optional crop coordinate
    :param w_start_at: optional crop coordinate
    :param w_end_at: optional crop coordinate
    :return: list of images corresponding to video frames
    """
    processed_frames = []
    for frame in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ret, image = cap.read()
        if ret:
            image = image[h_start_at:, w_start_at:w_end_at, :]  # crop x% of the video from the top
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
    batched_frames = [processed_frames[i:i + opt.fd_batch_size] for i in range(0, len(processed_frames), opt.fd_batch_size)]
    for frame_group in batched_frames:
        faces = face_detector(frame_group)
        all_faces += faces

    # threshold amount of faces, confidence level of 0.7
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
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = max(int(detections[0, 0, i, 3] * frameWidth), 0)  # left side of box
            y1 = max(int(detections[0, 0, i, 4] * frameHeight), 0)  # top side of box
            if x1 >= frameWidth or y1 >= frameHeight:  # if they are larger than image size, bbox is invalid
                continue
            x2 = min(int(detections[0, 0, i, 5] * frameWidth), frameWidth)  # either right side of box or frame width
            y2 = min(int(detections[0, 0, i, 6] * frameHeight), frameHeight)  # either the bottom side of box of frame height
            bboxes.append([x1, y1, x2-x1, y2-y1])  # (left, top, width, height)
    return bboxes
