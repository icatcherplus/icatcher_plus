import logging
import torch
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from . import options
from . import draw
from . import video
from . import models
from . import parsers
from . import version
import pooch

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


def select_face(bboxes, frame, fc_model, fc_data_transforms, hor, ver, device):
    """
    selects a correct face from candidates bbox in frame
    :param bboxes: the bounding boxes of candidates
    :param frame: the frame
    :param fc_model: a classifier model, if passed it is used to decide.
    :param fc_data_transforms: the transformations to apply to the images before fc_model sees them
    :param hor: the last known horizontal correct face location
    :param ver: the last known vertical correct face location
    :return: the cropped face and its bbox data
    """
    if fc_model:
        centers = []
        faces = []
        for box in bboxes:
            crop_img = frame[box[1]:box[1] + box[3], box[0]:box[0] + box[2]]
            face_box = np.array([box[1], box[1] + box[3], box[0], box[0] + box[2]])
            img_shape = np.array(frame.shape)
            ratio = np.array([face_box[0] / img_shape[0], face_box[1] / img_shape[0],
                              face_box[2] / img_shape[1], face_box[3] / img_shape[1]])
            face_ver = (ratio[0] + ratio[1]) / 2
            face_hor = (ratio[2] + ratio[3]) / 2

            centers.append([face_hor, face_ver])
            img = crop_img
            img = fc_data_transforms['val'](img)
            faces.append(img)
        centers = np.stack(centers)
        faces = torch.stack(faces).to(device)
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
    else:   # select lowest face in image, probably belongs to kid
        bbox = min(bboxes, key=lambda x: x[3] - x[1])
    return bbox

def fix_illegal_transitions(loc, answers, confidences, illegal_transitions, corrected_transitions):
    """
    this method fixes illegal transitions happening in answers at [loc-max_trans_len+1, loc] inclusive
    """
    for i, transition in enumerate(illegal_transitions):
        len_trans = len(transition)
        buffer = answers[loc-len_trans+1:loc+1]
        if buffer == transition:
            buffer_update = corrected_transitions[i]
            answers[loc-len_trans+1:loc+1] = buffer_update
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
    img_shape = np.array(frame.shape)
    face_box = np.array([bbox[1], bbox[1] + bbox[3], bbox[0], bbox[0] + bbox[2]])
    crop = frame[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]

    test_transforms = models.DataTransforms(opt.image_size,
                                          opt.per_channel_mean,
                                          opt.per_channel_std).transformations["test"]
    crop = test_transforms(Image.fromarray(crop))
    crop = crop.permute(1, 2, 0).unsqueeze(0).numpy()
    ratio = np.array([face_box[0] / img_shape[0], face_box[1] / img_shape[0],
                      face_box[2] / img_shape[1], face_box[3] / img_shape[1]])
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
        logging.warning("video file: {} has variable frame rate".format(str(video_path.name)))
        logging.info(str(meta_data))
        if opt.output_video_path:
            # todo: support this by extracting frame timestamps
            # i.e.: frame_info, vfr_frame_counter, _ = video.get_frame_information(video_path)
            logging.warning("output_video_path argument passed, but input video is VFR !")
    else:
        logging.info("video fps: {}".format(framerate))
    raw_width = meta_data["width"]
    raw_height = meta_data["height"]
    cropped_height = raw_height
    if "top" in opt.crop_mode:
        cropped_height = int(raw_height * (1 - (opt.crop_percent / 100)))  # x% of the video from the top
    cropped_width = raw_width
    if "left" and "right" in opt.crop_mode:
        cropped_width = int(raw_width * (1 - (2*opt.crop_percent / 100)))  # x% of the video from both left/right
    elif "left" in opt.crop_mode or "right" in opt.crop_mode:
        cropped_width = int(raw_width * (1 - (opt.crop_percent / 100)))  # x% of the video from both left/right
    resolution = (int(cropped_width), int(cropped_height))
    h_start_at = (raw_height - cropped_height)
    h_end_at = raw_height
    if "left" and "right" in opt.crop_mode:
        w_start_at = (raw_width - cropped_width)//2
        w_end_at = w_start_at + cropped_width
    elif "left" in opt.crop_mode:
        w_start_at = (raw_width - cropped_width)
        w_end_at = raw_width
    elif "right" in opt.crop_mode:
        w_start_at = 0
        w_end_at = cropped_width
    elif "top" in opt.crop_mode:
        w_start_at = 0
        w_end_at = raw_width
    return cap, framerate, resolution, h_start_at, h_end_at, w_start_at, w_end_at

def load_models(opt):
    """
    loads all relevant neural network models to perform predictions
    models will be automatically downloaded if not found in the cache,
    user may overide downloaded location with the env variable ICATCHER_DATA_DIR
    defaults:
    :Mac: "~/Library/Caches/<AppName>"
    :Unix: "~/.cache/<AppName>" or the value of the "XDG_CACHE_HOME"
    environment variable, if defined.
    :Windows: "C:\\Users\\<user>\\AppData\\Local\\<AppAuthor>\\<AppName>\\Cache"
    :param opt: command line options
    :return all nn models
    """
    GOODBOY = pooch.create(path=pooch.os_cache("icatcher_plus"),
                           base_url="https://osf.io/ycju8/download",
                           version=version,
                           version_dev="main",
                           env="ICATCHER_DATA_DIR",
                           registry={"zip_content.txt": "d81bfb5a183edea6dc74f7f342d516a9843865570b9ecfbf481209ec5114110a",
                                     "icatcher+_models.zip": "d78385b3a08f3d55ce75249142d15549e4c5552d5e1231cad3b69063bb778ce9"},
                           urls={"zip_content.txt":"https://osf.io/v4w53/download",
                                 "icatcher+_models.zip":"https://osf.io/ycju8/download"})
    # zip_content_file = GOODBOY.fetch("zip_content.txt")
    # with open(zip_content_file, "r") as f:
        # zip_content = [x.strip() for x in f]
    file_paths = GOODBOY.fetch("icatcher+_models.zip",
                               processor=pooch.Unzip(),
                               progressbar=True)
    file_names = [Path(x).name for x in file_paths]
    face_detector_model_file = file_paths[file_names.index("face_model.caffemodel")]
    config_file = file_paths[file_names.index("config.prototxt")]
    path_to_gaze_model = file_paths[file_names.index("icatcher+_lookit.pth")]
    if opt.model:
        path_to_gaze_model = opt.model
    path_to_fc_model = file_paths[file_names.index("face_classifier_lookit.pth")]
    if opt.fc_model:
        path_to_fc_model = opt.fc_model
    # face_detector_model_file = Path("models", "face_model.caffemodel")
    # config_file = Path("models", "config.prototxt")
    # path_to_gaze_model = opt.model
    gaze_model = models.GazeCodingModel(opt).to(opt.device)
    if opt.device == 'cpu':
        state_dict = torch.load(str(path_to_gaze_model), map_location=torch.device(opt.device))
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

    if opt.fc_model or opt.use_fc_model:
        face_classifier_model, fc_input_size = models.init_face_classifier(opt.device,
                                                                           num_classes=2,
                                                                           resume_from=path_to_fc_model)
        face_classifier_model.eval()
        face_classifier_model.to(opt.device)
        face_classifier_data_transforms = models.get_fc_data_transforms(fc_input_size)
    else:
        face_classifier_model = None
        face_classifier_data_transforms = None
    # load face extractor model
    face_detector_model = cv2.dnn.readNetFromCaffe(str(config_file), str(face_detector_model_file))    
    return gaze_model, face_detector_model, face_classifier_model, face_classifier_data_transforms

def get_video_paths(opt):
    """
    obtain the video paths (and possibly video ids) from the source argument
    :param opt: command line options
    :return: a list of video paths and a list of video ids
    """
    if opt.source_type == 'file':
        video_path = Path(opt.source)
        if video_path.is_dir():
            logging.warning("Video folder provided as source. Make sure it contains video files only.")
            video_paths = list(video_path.glob("*"))
            if opt.video_filter:
                filter_files = [x.stem for x in opt.video_filter.glob("*")]
                video_paths = [x for x in video_paths if x.stem in filter_files]
            video_paths = [str(x) for x in video_paths]
        elif video_path.is_file():
            video_paths = [str(video_path)]
        else:
            raise FileNotFoundError("Couldn't find a file or a directory at {}".format(video_path))
    else:
        # video_paths = [int(opt.source)]
        raise NotImplementedError
    return video_paths

def create_output_streams(video_path, framerate, resolution, opt):
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
    skip=False
    if opt.output_video_path:
        fourcc = cv2.VideoWriter_fourcc(*"MP4V")  # may need to be adjusted per available codecs & OS
        my_video_path = Path(opt.output_video_path, video_path.stem + "_output.mp4")
        video_output_file = cv2.VideoWriter(str(my_video_path), fourcc, framerate, resolution, True)
    if opt.output_annotation:
        if opt.output_format == "compressed":
            prediction_output_file = Path(opt.output_annotation, video_path.stem)
        else:
            prediction_output_file = Path(opt.output_annotation, video_path.stem + opt.output_file_suffix)
            if opt.output_format == "PrefLookTimestamp":
                with open(prediction_output_file, "w", newline="") as f: # Write header
                    f.write("Tracks: left, right, away, codingactive, outofframe\nTime,Duration,TrackName,comment\n\n")
    return video_output_file, prediction_output_file, skip
    
def cleanup(video_output_file, prediction_output_file, answers, confidences, framerate, frame_count, cap, opt):
    if opt.show_output:
        cv2.destroyAllWindows()
    if opt.output_video_path:
        video_output_file.release()
    if opt.output_annotation:  # write footer to file
        if opt.output_format == "PrefLookTimestamp":
            start_ms = int((1000. / framerate) * (opt.sliding_window_size // 2))
            end_ms = int((1000. / framerate) * frame_count)
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
    loc = -5  # where in the sliding window to take the prediction (should be a function of opt.sliding_window_size)
    cursor = -5 # points to the frame we will write to output relative to current frame
    classes = {'noface': -2, 'nobabyface': -1, 'away': 0, 'left': 1, 'right': 2}
    reverse_classes = {-2: 'noface', -1: 'nobabyface', 0: 'away', 1: 'left', 2: 'right'}
    logging.info("using the following values for per-channel mean: {}".format(opt.per_channel_mean))
    logging.info("using the following values for per-channel std: {}".format(opt.per_channel_std))
    gaze_model, face_detector_model, face_classifier_model, face_classifier_data_transforms = load_models(opt)
    video_paths = get_video_paths(opt)
    if opt.illegal_transitions_path:
        illegal_transitions, corrected_transitions = parsers.parse_illegal_transitions_file(opt.illegal_transitions_path)
        max_illegal_transition_length = max([len(transition) for transition in illegal_transitions])
        cursor -= max_illegal_transition_length
        if abs(cursor) > opt.sliding_window_size:
            raise ValueError("illegal_transitions_path contains transitions longer than the sliding window size")
    # loop over inputs
    for i in range(len(video_paths)):
        video_path = Path(str(video_paths[i]))
        logging.info("predicting on : {}".format(video_path))
        cap, framerate, resolution, h_start_at, h_end_at, w_start_at, w_end_at = process_video(video_path, opt)
        video_output_file, prediction_output_file, skip = create_output_streams(video_path, framerate, resolution, opt)
        if skip:
            continue
        # per video initialization
        answers = []  # list of answers for each frame
        confidences = []  # list of confidences for each frame
        image_sequence = []  # list of (crop, valid) for each frame in the sliding window
        box_sequence = []  # list of bounding boxes for each frame in the sliding window
        bbox_sequence = []  # list of bounding boxes for each frame in the sliding window
        frames = []  # list of frames for each frame in the sliding window
        from_tracker = []  # list of booleans indicating whether the bounding box was obtained from the tracker
        last_known_valid_bbox = None  # last known valid bounding box
        frame_count = 0  # frame counter
        hor, ver = 0.5, 1  # initial guess for face location
        cur_fps = video.FPS()  # for debugging purposes
        last_class_text = ""  # Initialize so that we see the first class assignment as an event to record
        # loop over frames (refactor !)
        ret_val, frame = cap.read()
        while ret_val:
            frame = draw.mask_regions(frame, h_start_at, h_end_at, w_start_at, w_end_at)  # mask roi
            # frame = frame[h_start_at:, w_start_at:w_end_at, :]  # crop x% of the video from the top
            frames.append(frame)
            cv2_bboxes = detect_face_opencv_dnn(face_detector_model, frame, 0.7)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # network was trained on RGB images.
            if not cv2_bboxes and (last_known_valid_bbox is None or not opt.track_face):
                answers.append(classes['noface'])  # if face detector fails, treat as away and mark invalid
                confidences.append(-1)
                image = np.zeros((1, opt.image_size, opt.image_size, 3), np.float64)
                my_box = np.array([0, 0, 0, 0, 0])
                image_sequence.append((image, True))
                box_sequence.append(my_box)
                bbox_sequence.append(None)
                from_tracker.append(False)
            else:
                if cv2_bboxes:
                    from_tracker.append(False)
                else:
                    from_tracker.append(True)
                    cv2_bboxes = [last_known_valid_bbox]
                selected_bbox = select_face(cv2_bboxes, frame, face_classifier_model, face_classifier_data_transforms, hor, ver, opt.device)
                crop, my_box = extract_crop(frame, selected_bbox, opt)
                if selected_bbox is None:
                    answers.append(classes['nobabyface'])  # if selecting face fails, treat as away and mark invalid
                    confidences.append(-1)
                    image = np.zeros((1, opt.image_size, opt.image_size, 3), np.float64)
                    my_box = np.array([0, 0, 0, 0, 0])
                    image_sequence.append((image, True))
                    box_sequence.append(my_box)
                    bbox_sequence.append(None)
                else:
                    assert crop.size != 0  # what just happened?
                    answers.append(classes['left'])  # if face detector succeeds, treat as left and mark valid
                    confidences.append(-1)
                    image_sequence.append((crop, False))
                    box_sequence.append(my_box)
                    bbox_sequence.append(selected_bbox)
                    if not from_tracker[-1]:
                        last_known_valid_bbox = selected_bbox.copy()
            if len(image_sequence) == opt.sliding_window_size:  # we have enough frames for prediction, predict for middle frame
                cur_frame = frames[cursor]
                cur_bbox = bbox_sequence[cursor]
                is_from_tracker = from_tracker[cursor]
                frames.pop(0)
                bbox_sequence.pop(0)
                from_tracker.pop(0)
                if not image_sequence[opt.sliding_window_size // 2][1]:  # if middle image is valid
                    to_predict = {"imgs": torch.tensor(np.array([x[0] for x in image_sequence[0::2]]), dtype=torch.float).squeeze().permute(0, 3, 1, 2).to(opt.device),
                                    "boxs": torch.tensor(np.array(box_sequence[::2]), dtype=torch.float).to(opt.device)
                                    }
                    with torch.set_grad_enabled(False):
                        outputs = gaze_model(to_predict)  # actual gaze prediction
                        probs = torch.nn.functional.softmax(outputs, dim=1)
                        _, prediction = torch.max(outputs, 1)
                        confidence, _ = torch.max(probs, 1)
                        float32_conf = confidence.cpu().numpy()[0]
                        int32_pred = prediction.cpu().numpy()[0]
                    answers[loc] = int32_pred  # update answers for the middle frame
                    confidences[loc] = float32_conf  # update confidences for the middle frame
                image_sequence.pop(0)
                box_sequence.pop(0)

                if opt.illegal_transitions_path:
                    if len(answers) >= max_illegal_transition_length: 
                        answers, confidences = fix_illegal_transitions(loc, answers, confidences, illegal_transitions, corrected_transitions)
                class_text = reverse_classes[answers[cursor]]
                if opt.on_off:
                    class_text = "off" if class_text == "away" else "on"
                if opt.output_video_path:
                    if is_from_tracker and opt.track_face:
                        rect_color = (0, 0, 255)
                    else:
                        rect_color = (0, 255, 0) 
                    draw.prepare_frame(cur_frame, cur_bbox, show_arrow=True, rect_color=rect_color,
                                         conf=confidences[cursor], class_text=class_text,
                                         frame_number=frame_count, pic_in_pic=opt.pic_in_pic)
                    video_output_file.write(cur_frame)
                if opt.show_output:
                    if is_from_tracker and opt.track_face:
                        rect_color = (0, 0, 255)
                    else:
                        rect_color = (0, 255, 0) 
                    draw.prepare_frame(cur_frame, cur_bbox, show_arrow=True, rect_color=rect_color,
                                         conf=confidences[cursor], class_text=class_text,
                                         frame_number=frame_count, pic_in_pic=opt.pic_in_pic)
                    
                    cv2.imshow('frame', cur_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                # handle writing output to file
                if opt.output_annotation:
                    if opt.output_format == "raw_output":
                        with open(prediction_output_file, "a", newline="") as f:
                            f.write("{}, {}, {:.02f}\n".format(str(frame_count + cursor + 1), class_text, confidences[cursor]))
                    elif opt.output_format == "PrefLookTimestamp":
                        if class_text != last_class_text:  # Record "event" for change of direction if code has changed
                            frame_ms = int((frame_count + cursor + 1) * (1000. / framerate))
                            with open(prediction_output_file, "a", newline="") as f:
                                f.write("{},0,{}\n".format(frame_ms, class_text))
                            last_class_text = class_text
                logging.info("frame: {}, class: {}, confidence: {:.02f}, cur_fps: {:.02f}".format(str(frame_count + cursor + 1), class_text, confidences[cursor], cur_fps()))
            ret_val, frame = cap.read()
            frame_count += 1
        # finished processing a video file, cleanup
        cleanup(video_output_file, prediction_output_file, answers, confidences, framerate, frame_count, cap, opt)

def main():
    args = options.parse_arguments()
    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(filename=args.log, filemode='w', level=args.verbosity.upper())
    else:
        logging.basicConfig(level=args.verbosity.upper())
    predict_from_video(args)

if __name__ == "__main__":
    main()