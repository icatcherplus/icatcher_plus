import cv2
from PIL import Image
from pathlib import Path
import numpy as np
from preprocess import build_lookit_video_dataset, build_marchman_video_dataset
import options
import visualize
import logging
import face_classifier
import torch
import face_recognition
import models
import data
import video
import time
import collections
from parsers import parse_illegal_transitions_file
from face_detector import extract_bboxes, process_frames, parallelize_face_detection, detect_face_opencv_dnn, create_retina_model
from face_detection import RetinaFace
import multiprocessing as mp
from face_rec import FaceRec


class FPS:
    """
    calculates current fps and returns it, see https://stackoverflow.com/a/54539292
    """

    def __init__(self, avarageof=50):
        self.frametimestamps = collections.deque(maxlen=avarageof)

    def __call__(self):
        self.frametimestamps.append(time.time())
        if len(self.frametimestamps) > 1:
            return len(self.frametimestamps) / (
                self.frametimestamps[-1] - self.frametimestamps[0]
            )
        else:
            return 0.0


class FaceClassifierArgs:
    """
    encapsulates face classifier arguments
    """

    def __init__(self, device):
        self.device = device
        self.rotation = False
        self.cropping = False
        self.hor_flip = False
        self.ver_flip = False
        self.color = False
        self.erasing = False
        self.noise = False
        self.model = "vgg16"
        self.dropout = 0.0


def select_face(bboxes, frame, fc_model, fc_data_transforms, hor, ver, opt):
    """
    selects a correct face from candidates bbox in frame
    :param bboxes: the bounding boxes of candidates
    :param frame: the frame
    :param fc_model: a classifier model, if passed it is used to decide.
    :param fc_data_transforms: the transformations to apply to the images before fc_model sees them
    :param hor: the last known horizontal correct face location
    :param ver: the last known vertical correct face location
    :param opt: used to pull the device id
    :param fr: face recognition, used only in the case where it's flagged and no faces were found selected
    :return: the cropped face and its bbox data
    """
    if fc_model:
        centers = []
        faces = []
        for box in bboxes:
            crop_img = frame[box[1] : box[1] + box[3], box[0] : box[0] + box[2]]
            face_box = np.array([box[1], box[1] + box[3], box[0], box[0] + box[2]])
            img_shape = np.array(frame.shape)
            ratio = np.array(
                [
                    face_box[0] / img_shape[0],
                    face_box[1] / img_shape[0],
                    face_box[2] / img_shape[1],
                    face_box[3] / img_shape[1],
                ]
            )
            face_ver = (ratio[0] + ratio[1]) / 2
            face_hor = (ratio[2] + ratio[3]) / 2

            centers.append([face_hor, face_ver])
            img = crop_img
            img = fc_data_transforms["val"](img)
            faces.append(img)
        centers = np.stack(centers)
        faces = torch.stack(faces).to(opt.device)
        output = fc_model(faces)
        _, preds = torch.max(output, 1)
        preds = preds.cpu().numpy()
        idxs = np.where(preds == 0)[0]
        if idxs.size == 0:
            bbox = None
        else:
            centers = centers[idxs]
            dis = np.sqrt((centers[:, 0] - hor) ** 2 + (centers[:, 1] - ver) ** 2)
            i = np.argmin(dis)
            # crop_img = faces[idxs[i]]
            bbox = bboxes[idxs[i]]
            # hor, ver = centers[i]
    else:  # select lowest face in image, probably belongs to kid
        bbox = min(bboxes, key=lambda x: x[3] - x[1])
    return bbox


def fix_illegal_transitions(loc, answers, confidences, illegal_transitions, corrected_transitions):
    """
    this method fixes illegal transitions happening in answers at [loc-max_trans_len+1, loc] inclusive
    """
    for i, transition in enumerate(illegal_transitions):
        len_trans = len(transition)
        buffer = answers[loc - len_trans + 1 : loc + 1]
        if buffer == transition:
            buffer_update = corrected_transitions[i]
            answers[loc - len_trans + 1 : loc + 1] = buffer_update
            buffer_splits = np.where(np.array(buffer_update) != np.array(buffer))
            for spot in buffer_splits[0].tolist():
                confidences[loc - len_trans + 1 + spot] = -1
    return answers, confidences


def extract_crop(frame, bbox, opt):
    """
    extracts a crop from a frame using bbox, and transforms it
    :param frame: the frame
    :param bbox: opencv bbox 4x1
    :param opt: command line options
    :return: the crop and the 5x1 box features
    """
    if bbox is None:
        return None, None

    # make sure no negatives being fed into extract crop
    bbox = [0 if x < 0 else x for x in bbox]

    img_shape = np.array(frame.shape)
    face_box = np.array([bbox[1], bbox[1] + bbox[3], bbox[0], bbox[0] + bbox[2]])
    crop = frame[bbox[1] : bbox[1] + bbox[3], bbox[0] : bbox[0] + bbox[2]]

    test_transforms = data.DataTransforms(
        opt.image_size, opt.per_channel_mean, opt.per_channel_std
    ).transformations["test"]
    crop = test_transforms(Image.fromarray(crop))
    crop = crop.permute(1, 2, 0).unsqueeze(0).numpy()
    ratio = np.array(
        [
            face_box[0] / img_shape[0],
            face_box[1] / img_shape[0],
            face_box[2] / img_shape[1],
            face_box[3] / img_shape[1],
        ]
    )
    face_size = (ratio[1] - ratio[0]) * (ratio[3] - ratio[2])
    face_ver = (ratio[0] + ratio[1]) / 2
    face_hor = (ratio[2] + ratio[3]) / 2
    face_height = ratio[1] - ratio[0]
    face_width = ratio[3] - ratio[2]
    my_box = np.array([face_size, face_ver, face_hor, face_height, face_width])
    return crop, my_box


def process_video(video_path, opt):
    """
    give a video path, process it and return a generator to iterate over frames
    :param video_path: the video path
    :param opt: command line options
    :return: a generator to iterate over frames, framerate, resolution, and height/width pixel coordinates to crop from
    """
    cap = cv2.VideoCapture(str(video_path))
    # Get some basic info about the video
    vfr, meta_data = video.is_video_vfr(video_path, get_meta_data=True)
    framerate = video.get_fps(video_path)
    if vfr:
        logging.warning(
            "video file: {} has variable frame rate".format(str(video_path.name))
        )
        logging.info(str(meta_data))
        if opt.output_video_path:
            # todo: support this by extracting frame timestamps
            # i.e.: frame_info, vfr_frame_counter, _ = video.get_frame_information(video_path)
            logging.warning(
                "output_video_path argument passed, but input video is VFR !"
            )
    else:
        logging.info("video fps: {}".format(framerate))
    raw_width = meta_data["width"]
    raw_height = meta_data["height"]
    cropped_height = raw_height
    if "top" in opt.crop_mode:
        cropped_height = int(
            raw_height * (1 - (opt.crop_percent / 100))
        )  # crop x% of the video from the top
    cropped_width = raw_width
    if "left" and "right" in opt.crop_mode:
        cropped_width = int(
            raw_width * (1 - (2 * opt.crop_percent / 100))
        )  # crop x% of the video from the top
    elif "left" in opt.crop_mode or "right" in opt.crop_mode:
        cropped_width = int(raw_width * (1 - (opt.crop_percent / 100)))
    resolution = (int(cropped_width), int(cropped_height))
    h_start_at = raw_height - cropped_height
    if "left" and "right" in opt.crop_mode:
        w_start_at = (raw_width - cropped_width) // 2
        w_end_at = w_start_at + cropped_width
    elif "left" in opt.crop_mode:
        w_start_at = raw_width - cropped_width
        w_end_at = raw_width
    elif "right" in opt.crop_mode:
        w_start_at = 0
        w_end_at = cropped_width
    elif "top" in opt.crop_mode:
        w_start_at = 0
        w_end_at = raw_width
    return cap, framerate, resolution, h_start_at, w_start_at, w_end_at


def load_models(opt):
    """
    loads all relevant neural network models to perform predictions
    :param opt: command line options
    :return all nn models
    """
    if opt.fd_model == "retinaface":  # option for retina face vs. previous opencv dnn model
        face_detector_model = create_retina_model(gpu_id=opt.gpu_id)
    elif opt.fd_model == "opencv_dnn":
        face_detector_model_file = Path("models", "face_model.caffemodel")
        config_file = Path("models", "config.prototxt")
        face_detector_model = cv2.dnn.readNetFromCaffe(str(config_file), str(face_detector_model_file))
    else:
        raise NotImplementedError
    path_to_gaze_model = opt.model
    if opt.architecture == "icatcher+":
        gaze_model = models.GazeCodingModel(opt).to(opt.device)
    elif opt.architecture == "icatcher_vanilla":
        gaze_model = models.iCatcherOriginal(opt).to(opt.device)
    else:
        raise NotImplementedError
    if opt.device == "cpu":
        state_dict = torch.load(
            str(path_to_gaze_model), map_location=torch.device(opt.device)
        )
    else:
        state_dict = torch.load(str(path_to_gaze_model))
    try:
        gaze_model.load_state_dict(state_dict)
    except RuntimeError as e:  # hack to deal with models trained on distributed setup
        from collections import OrderedDict

        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        # load params
        gaze_model.load_state_dict(new_state_dict)
    gaze_model.eval()

    if opt.fc_model:
        fc_args = FaceClassifierArgs(opt.device)
        (
            face_classifier_model,
            fc_input_size,
        ) = face_classifier.fc_model.init_face_classifier(
            fc_args, model_name=fc_args.model, num_classes=2, resume_from=opt.fc_model
        )
        face_classifier_model.eval()
        face_classifier_model.to(opt.device)
        face_classifier_data_transforms = (
            face_classifier.fc_eval.get_fc_data_transforms(fc_args, fc_input_size)
        )
    else:
        face_classifier_model = None
        face_classifier_data_transforms = None
    return gaze_model, face_detector_model, face_classifier_model, face_classifier_data_transforms

def get_video_paths(opt):
    """
    obtain the video paths (and possibly video ids) from the source argument
    :param opt: command line options
    :return: a list of video paths and a list of video ids
    """
    if opt.source_type == "file":
        video_path = Path(opt.source)
        video_ids = None
        if video_path.is_dir():
            logging.warning(
                "Video folder provided as source. Make sure it contains video files only."
            )
            video_paths = list(video_path.glob("*"))
            if opt.video_filter:
                if opt.video_filter.is_file():
                    if opt.raw_dataset_type == "lookit":
                        video_dataset = build_lookit_video_dataset(
                            opt.raw_dataset_path, opt.video_filter
                        )
                    elif (
                        opt.raw_dataset_type == "cali-bw"
                        or opt.raw_dataset_type == "senegal"
                    ):
                        video_dataset = build_marchman_video_dataset(
                            opt.raw_dataset_path, opt.raw_dataset_type
                        )
                    else:
                        raise NotImplementedError
                    # filter_files = [x for x in video_dataset.values() if
                    #                 x["in_csv"] and x["has_1coding"] and x["public"]]
                    # filter_files = [x for x in video_dataset.values() if x["video_id"] == ""]
                    filter_files = [
                        x
                        for x in video_dataset.values()
                        if x["in_csv"]
                        and x["has_1coding"]
                        and x["has_2coding"]
                        and x["split"] == "2_test"
                    ]
                    video_ids = [x["video_id"] for x in filter_files]
                    filter_files = [x["video_path"].stem for x in filter_files]
                else:  # directory
                    filter_files = [x.stem for x in opt.video_filter.glob("*")]
                video_paths = [x for x in video_paths if x.stem in filter_files]
            video_paths = [str(x) for x in video_paths]
        elif video_path.is_file():
            video_paths = [str(video_path)]
        else:
            raise FileNotFoundError
    else:
        # video_paths = [int(opt.source)]
        raise NotImplementedError
    return video_paths, video_ids


def create_output_streams(video_path, framerate, resolution, video_ids, opt):
    """
    creates output streams
    :param video_path: path to video
    :param framerate: video framerate
    :param resolution: video resolution
    :param video_ids: list of video ids
    :param opt: options
    :return: video_output_file, prediction_output_file, skip = prediction file already exists
    """
    video_output_file = None
    prediction_output_file = None
    skip = False
    if opt.output_video_path:
        fourcc = cv2.VideoWriter_fourcc(
            *"MP4V"
        )  # may need to be adjusted per available codecs & OS
        my_video_path = Path(opt.output_video_path, video_path.stem + "_output.mp4")
        video_output_file = cv2.VideoWriter(
            str(my_video_path), fourcc, framerate, resolution, True
        )
    if opt.output_annotation:
        if opt.output_format == "compressed":
            if video_ids is not None:
                prediction_output_file = Path(opt.output_annotation, video_ids[i])
                if Path(str(prediction_output_file) + ".npz").is_file():
                    skip = True
            else:
                prediction_output_file = Path(opt.output_annotation, video_path.stem)
        else:
            prediction_output_file = Path(
                opt.output_annotation, video_path.stem + opt.output_file_suffix
            )
            if opt.output_format == "PrefLookTimestamp":
                with open(prediction_output_file, "w", newline="") as f:  # Write header
                    f.write(
                        "Tracks: left, right, away, codingactive, outofframe\nTime,Duration,TrackName,comment\n\n"
                    )
    return video_output_file, prediction_output_file, skip


def cleanup(video_output_file, prediction_output_file, answers, confidences, framerate, frame_count, cap, opt):
    if opt.show_output:
        cv2.destroyAllWindows()
    if opt.output_video_path:
        video_output_file.release()
    if opt.output_annotation:  # write footer to file
        if opt.output_format == "PrefLookTimestamp":
            start_ms = int((1000.0 / framerate) * (opt.sliding_window_size // 2))
            end_ms = int((1000.0 / framerate) * frame_count)
            with open(prediction_output_file, "a", newline="") as f:
                f.write("{},{},codingactive\n".format(start_ms, end_ms))
        elif opt.output_format == "compressed":
            np.savez(prediction_output_file, answers, confidences)
    cap.release()


def predict_from_video(opt):
    """
    perform prediction on a stream or video file(s) using a network.
    output can be of various kinds, see options for details.
    :param opt: command line arguments
    :return:
    """
    # initialize
    loc = (
        -5
    )  # where in the sliding window to take the prediction (should be a function of opt.sliding_window_size)
    cursor = -5  # points to the frame we will write to output relative to current frame
    classes = {"noface": -2, "nobabyface": -1, "away": 0, "left": 1, "right": 2}
    reverse_classes = {-2: "noface", -1: "nobabyface", 0: "away", 1: "left", 2: "right"}
    logging.info(
        "using the following values for per-channel mean: {}".format(
            opt.per_channel_mean
        )
    )
    logging.info(
        "using the following values for per-channel std: {}".format(opt.per_channel_std)
    )
    (
        gaze_model,
        face_detector_model,
        face_classifier_model,
        face_classifier_data_transforms,
    ) = load_models(opt)
    video_paths, video_ids = get_video_paths(opt)
    if opt.illegal_transitions_path:
        illegal_transitions, corrected_transitions = parse_illegal_transitions_file(
            opt.illegal_transitions_path
        )
        max_illegal_transition_length = max(
            [len(transition) for transition in illegal_transitions]
        )
        cursor -= max_illegal_transition_length
        if abs(cursor) > opt.sliding_window_size:
            raise ValueError("illegal_transitions_path contains transitions longer than the sliding window size")
    # check if cpu or gpu being used
    use_cpu = True if opt.gpu_id == -1 else False
    # loop over inputs
    for i in range(len(video_paths)):
        video_path = Path(str(video_paths[i]))
        logging.info("predicting on : {}".format(video_path))
        cap, framerate, resolution, h_start_at, w_start_at, w_end_at = process_video(
            video_path, opt
        )
        video_output_file, prediction_output_file, skip = create_output_streams(
            video_path, framerate, resolution, video_ids, opt
        )
        if skip:
            continue
        # per video initialization
        answers = []  # list of answers for each frame
        confidences = []  # list of confidences for each frame
        image_sequence = (
            []
        )  # list of (crop, valid) for each frame in the sliding window
        box_sequence = []  # list of bounding boxes for each frame in the sliding window
        bbox_sequence = (
            []
        )  # list of bounding boxes for each frame in the sliding window
        frames = []  # list of frames for each frame in the sliding window
        from_tracker = (
            []
        )  # list of booleans indicating whether the bounding box was obtained from the tracker
        last_known_valid_bbox = None  # last known valid bounding box
        frame_count = 0  # frame counter
        hor, ver = 0.5, 1  # initial guess for face location
        cur_fps = FPS()  # for debugging purposes
        last_class_text = ""  # Initialize so that we see the first class assignment as an event to record

        # if going to use cpu parallelization, don't allow for live stream video
        if use_cpu and opt.fd_model == "retinaface":  # TODO: add output timing feature to show fps processing
            # figure out how many cpus can be used
            num_cpus = mp.cpu_count() - opt.num_cpus_saved
            assert num_cpus > 0

            # send all frames in to be preprocessed and have faces detected prior to running gaze detection
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            vid_frames = range(0, total_frames, 1 + opt.fd_skip_frames)  # adding step if frames are skipped
            processed_frames = process_frames(cap, vid_frames, h_start_at, w_start_at, w_end_at)
            faces = parallelize_face_detection(processed_frames, face_detector_model, num_cpus, opt)
            del processed_frames

            # flatten the list and extract bounding boxes
            faces = [item for sublist in faces for item in sublist]
            master_bboxes = [extract_bboxes(face_group) for face_group in faces]

            # if frames were skipped, need to repeat binding boxes for that many skips
            if opt.fd_skip_frames > 0:
                master_bboxes = np.repeat(master_bboxes, opt.fd_skip_frames + 1)

            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # reset frames to 0

        #Set up face recognition for use
        if opt.use_facerec != None:
            fr = FaceRec()
            if opt.facerec == "reference":
                fr.get_ref_image(opt.facerec_ref)

        # loop over frames
        ret_val, frame = cap.read()
        while frame is not None:
            frame = frame[h_start_at:, w_start_at:w_end_at, :]  # crop x% of the video from the top
            frames.append(frame)

            if use_cpu and opt.fd_model == "retinaface":  # if using cpu, just pull from master
                bboxes = master_bboxes[frame_count]
            elif opt.fd_model == "opencv_dnn":
                bboxes = detect_face_opencv_dnn(face_detector_model, frame, opt.fd_confidence_threshold)
            else:  # if using gpu, able to find face as frame is processed... don't need batch inference
                faces = face_detector_model(frame)
                faces = [face for face in faces if face[-1] >= opt.fd_confidence_threshold]
                bboxes = extract_bboxes(faces)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # network was trained on RGB images.
            # if len(cv2_bboxes) > 2:
            #     visualize.temp_hook(frame, cv2_bboxes, frame_count)
            if not bboxes and (last_known_valid_bbox is None or not opt.track_face):
                answers.append(classes['noface'])  # if face detector fails, treat as away and mark invalid
                confidences.append(-1)
                image = np.zeros((1, opt.image_size, opt.image_size, 3), np.float64)
                my_box = np.array([0, 0, 0, 0, 0])
                image_sequence.append((image, True))
                box_sequence.append(my_box)
                bbox_sequence.append(None)
                from_tracker.append(False)
            else:
                if bboxes:
                    from_tracker.append(False)
                else:
                    from_tracker.append(True)
                    bboxes = [last_known_valid_bbox]
                
                if opt.use_facerec and last_known_valid_bbox: #Only use facerec if its ready
                    if opt.facerec == "bbox" and len(fr.known_faces) == 0: #If no known faces, generate a reference image
                        
                        fr.generate_ref_image(fr.convert_bounding_boxes([last_known_valid_bbox]), frame)
                    selected_bbox = fr.select_face(bboxes, frame)
                else:
                    selected_bbox = select_face(bboxes, frame, face_classifier_model, face_classifier_data_transforms, hor, ver, opt)
                
                crop, my_box = extract_crop(frame, selected_bbox, opt)
                if selected_bbox is None:
                    answers.append(
                        classes["nobabyface"]
                    )  # if selecting face fails, treat as away and mark invalid
                    confidences.append(-1)
                    image = np.zeros((1, opt.image_size, opt.image_size, 3), np.float64)
                    my_box = np.array([0, 0, 0, 0, 0])
                    image_sequence.append((image, True))
                    box_sequence.append(my_box)
                    bbox_sequence.append(None)
                else:
                    assert crop.size != 0  # what just happened?
                    answers.append(
                        classes["left"]
                    )  # if face detector succeeds, treat as left and mark valid
                    confidences.append(-1)
                    image_sequence.append((crop, False))
                    box_sequence.append(my_box)
                    bbox_sequence.append(selected_bbox)
                    if not from_tracker[-1]:
                        last_known_valid_bbox = selected_bbox.copy()
            if (
                len(image_sequence) == opt.sliding_window_size
            ):  # we have enough frames for prediction, predict for middle frame
                cur_frame = frames[cursor]
                cur_bbox = bbox_sequence[cursor]
                is_from_tracker = from_tracker[cursor]
                frames.pop(0)
                bbox_sequence.pop(0)
                from_tracker.pop(0)
                if not image_sequence[opt.sliding_window_size // 2][
                    1
                ]:  # if middle image is valid
                    if opt.architecture == "icatcher+":
                        to_predict = {
                            "imgs": torch.tensor(
                                np.array([x[0] for x in image_sequence[0::2]]),
                                dtype=torch.float,
                            )
                            .squeeze()
                            .permute(0, 3, 1, 2)
                            .to(opt.device),
                            "boxs": torch.tensor(
                                np.array(box_sequence[::2]), dtype=torch.float
                            ).to(opt.device),
                        }
                    elif opt.architecture == "icatcher_vanilla":
                        to_predict = {
                            "imgs": torch.tensor(
                                [x[0] for x in image_sequence[0::2]], dtype=torch.float
                            )
                            .permute(1, 0, 4, 2, 3)
                            .to(opt.device)
                        }
                    else:
                        raise NotImplementedError
                    with torch.set_grad_enabled(False):
                        outputs = gaze_model(to_predict)  # actual gaze prediction
                        probs = torch.nn.functional.softmax(outputs, dim=1)
                        _, prediction = torch.max(outputs, 1)
                        confidence, _ = torch.max(probs, 1)
                        print(confidence)
                        float32_conf = confidence.cpu().numpy()[0]
                        int32_pred = prediction.cpu().numpy()[0]
                    answers[loc] = int32_pred  # update answers for the middle frame
                    confidences[
                        loc
                    ] = float32_conf  # update confidences for the middle frame
                image_sequence.pop(0)
                box_sequence.pop(0)

                if opt.illegal_transitions_path:
                    if len(answers) >= max_illegal_transition_length:
                        answers, confidences = fix_illegal_transitions(
                            loc,
                            answers,
                            confidences,
                            illegal_transitions,
                            corrected_transitions,
                        )
                class_text = reverse_classes[answers[cursor]]
                if opt.on_off:
                    class_text = "off" if class_text == "away" else "on"
                if opt.output_video_path:
                    if is_from_tracker and opt.track_face:
                        rect_color = (0, 0, 255)
                    else:
                        rect_color = (0, 255, 0)
                    visualize.prep_frame(
                        cur_frame,
                        cur_bbox,
                        show_arrow=True,
                        rect_color=rect_color,
                        conf=confidences[cursor],
                        class_text=class_text,
                        frame_number=frame_count,
                        pic_in_pic=opt.pic_in_pic,
                    )
                    video_output_file.write(cur_frame)
                if opt.show_output:
                    if is_from_tracker and opt.track_face:
                        rect_color = (0, 0, 255)
                    else:
                        rect_color = (0, 255, 0)
                    visualize.prep_frame(
                        cur_frame,
                        cur_bbox,
                        show_arrow=True,
                        rect_color=rect_color,
                        conf=confidences[cursor],
                        class_text=class_text,
                        frame_number=frame_count,
                        pic_in_pic=opt.pic_in_pic,
                    )

                    cv2.imshow("frame", cur_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                # handle writing output to file
                if opt.output_annotation:
                    if opt.output_format == "raw_output":
                        with open(prediction_output_file, "a", newline="") as f:
                            f.write(
                                "{}, {}, {:.02f}\n".format(
                                    str(frame_count + cursor + 1),
                                    class_text,
                                    confidences[cursor],
                                )
                            )
                    elif opt.output_format == "PrefLookTimestamp":
                        if (
                            class_text != last_class_text
                        ):  # Record "event" for change of direction if code has changed
                            frame_ms = int(
                                (frame_count + cursor + 1) * (1000.0 / framerate)
                            )
                            with open(prediction_output_file, "a", newline="") as f:
                                f.write("{},0,{}\n".format(frame_ms, class_text))
                            last_class_text = class_text
                logging.info(
                    "frame: {}, class: {}, confidence: {:.02f}, cur_fps: {:.02f}".format(
                        str(frame_count + cursor + 1),
                        class_text,
                        confidences[cursor],
                        cur_fps(),
                    )
                )
            ret_val, frame = cap.read()
            frame_count += 1
        # finished processing a video file, cleanup
        cleanup(
            video_output_file,
            prediction_output_file,
            answers,
            confidences,
            framerate,
            frame_count,
            cap,
            opt,
        )


if __name__ == "__main__":
    args = options.parse_arguments_for_testing()
    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            filename=args.log, filemode="w", level=args.verbosity.upper()
        )
    else:
        logging.basicConfig(level=args.verbosity.upper())
    predict_from_video(args)
