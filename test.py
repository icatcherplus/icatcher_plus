import cv2
from PIL import Image
from pathlib import Path
import numpy as np
from preprocess import detect_face_opencv_dnn, build_lookit_video_dataset, build_marchman_video_dataset
import options
import visualize
import logging
import face_classifier
import torch
import models
import data
import video
import time
import collections


class FPS:
    """
    calculates current fps and returns it, see https://stackoverflow.com/a/54539292
    """
    def __init__(self,avarageof=50):
        self.frametimestamps = collections.deque(maxlen=avarageof)
    def __call__(self):
        self.frametimestamps.append(time.time())
        if(len(self.frametimestamps) > 1):
            return len(self.frametimestamps)/(self.frametimestamps[-1]-self.frametimestamps[0])
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


def select_face(bboxes, frame, fc_model, fc_data_transforms, hor, ver):
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
        faces = torch.stack(faces).to(args.device)
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

    test_transforms = data.DataTransforms(opt.image_size,
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


def predict_from_video(opt):
    """
    perform prediction on a stream or video file(s) using a network.
    output can be of various kinds, see options for details.
    :param opt:
    :return:
    """
    # todo: refactor, this function is too big
    # initialize
    opt.sliding_window_size = 9
    opt.window_stride = 2
    loc = -5
    classes = {'noface': -2, 'nobabyface': -1, 'away': 0, 'left': 1, 'right': 2}
    reverse_classes = {-2: 'noface', -1: 'nobabyface', 0: 'away', 1: 'left', 2: 'right'}
    logging.info("using the following values for per-channel mean: {}".format(opt.per_channel_mean))
    logging.info("using the following values for per-channel std: {}".format(opt.per_channel_std))
    face_detector_model_file = Path("models", "face_model.caffemodel")
    config_file = Path("models", "config.prototxt")
    path_to_primary_model = opt.model
    if opt.architecture == "icatcher+":
        primary_model = models.GazeCodingModel(opt).to(opt.device)
    elif opt.architecture == "icatcher_vanilla":
        primary_model = models.iCatcherOriginal(opt).to(opt.device)
    else:
        raise NotImplementedError
    if opt.device == 'cpu':
        state_dict = torch.load(str(path_to_primary_model), map_location=torch.device(opt.device))
    else:
        state_dict = torch.load(str(path_to_primary_model))
    try:
        primary_model.load_state_dict(state_dict)
    except RuntimeError as e:  # deal with models trained on distributed setup
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        # load params
        primary_model.load_state_dict(new_state_dict)
    primary_model.eval()

    if opt.fc_model:
        fc_args = FaceClassifierArgs(opt.device)
        fc_model, fc_input_size = face_classifier.fc_model.init_face_classifier(fc_args,
                                                                                model_name=fc_args.model,
                                                                                num_classes=2,
                                                                                resume_from=opt.fc_model)
        fc_model.eval()
        fc_model.to(opt.device)
        fc_data_transforms = face_classifier.fc_eval.get_fc_data_transforms(fc_args,
                                                                            fc_input_size)
    else:
        fc_model = None
        fc_data_transforms = None
    # load face extractor model
    face_detector_model = cv2.dnn.readNetFromCaffe(str(config_file), str(face_detector_model_file))
    # set video source
    if opt.source_type == 'file':
        video_path = Path(opt.source)
        video_ids = None
        if video_path.is_dir():
            logging.warning("Video folder provided as source. Make sure it contains video files only.")
            video_paths = list(video_path.glob("*"))
            if opt.video_filter:
                if opt.video_filter.is_file():
                    if opt.raw_dataset_type == "lookit":
                        video_dataset = build_lookit_video_dataset(opt.raw_dataset_path, opt.video_filter)
                    elif opt.raw_dataset_type == "cali-bw" or opt.raw_dataset_type == "senegal":
                        video_dataset = build_marchman_video_dataset(opt.raw_dataset_path, opt.raw_dataset_type)
                    else:
                        raise NotImplementedError
                    # filter_files = [x for x in video_dataset.values() if
                    #                 x["in_csv"] and x["has_1coding"] and x["public"]]
                    # filter_files = [x for x in video_dataset.values() if x["video_id"] == ""]
                    filter_files = [x for x in video_dataset.values() if
                                    x["in_csv"] and x["has_1coding"] and x["has_2coding"] and x[
                                        "split"] == "2_test"]
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
    for i in range(len(video_paths)):
        video_path = Path(str(video_paths[i]))
        answers = []
        confidences = []
        image_sequence = []
        box_sequence = []
        bbox_sequence = []
        frames = []
        frame_count = 0
        cur_fps = FPS()
        last_class_text = ""  # Initialize so that we see the first class assignment as an event to record
        logging.info("predicting on : {}".format(video_paths[i]))
        cap = cv2.VideoCapture(video_paths[i])
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
        width = meta_data["width"]
        height = meta_data["height"]
        resolution = (int(width), int(height))
        # If creating annotated video output, set up now
        if opt.output_video_path:
            fourcc = cv2.VideoWriter_fourcc(*"MP4V")  # may need to be adjusted per available codecs & OS
            my_video_path = Path(opt.output_video_path, video_path.stem + "_output.mp4")
            video_output = cv2.VideoWriter(str(my_video_path), fourcc, framerate, resolution, True)
        if opt.output_annotation:
            if opt.output_format == "compressed":
                if video_ids is not None:
                    my_output_file_path = Path(opt.output_annotation, video_ids[i])
                    if Path(str(my_output_file_path) + ".npz").is_file():
                        continue
                else:
                    my_output_file_path = Path(opt.output_annotation, video_path.stem)
            else:
                my_output_file_path = Path(opt.output_annotation, video_path.stem + opt.output_file_suffix)
                output_file = open(my_output_file_path, "w", newline="")
            if opt.output_format == "PrefLookTimestamp":
                # Write header
                output_file.write(
                    "Tracks: left, right, away, codingactive, outofframe\nTime,Duration,TrackName,comment\n\n")
        # iterate over frames
        ret_val, frame = cap.read()
        hor, ver = 0.5, 1  # used for improved selection of face
        while ret_val:
            frames.append(frame)
            cv2_bboxes = detect_face_opencv_dnn(face_detector_model, frame, 0.7)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # network was trained on RGB images.
            # if len(cv2_bboxes) > 2:
            #     visualize.temp_hook(frame, cv2_bboxes, frame_count)
            if not cv2_bboxes:
                answers.append(classes['noface'])  # if face detector fails, treat as away and mark invalid
                confidences.append(-1)
                image = np.zeros((1, opt.image_size, opt.image_size, 3), np.float64)
                my_box = np.array([0, 0, 0, 0, 0])
                image_sequence.append((image, True))
                box_sequence.append(my_box)
                bbox_sequence.append(None)
            else:
                selected_bbox = select_face(cv2_bboxes, frame, fc_model, fc_data_transforms, hor, ver)
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
                    hor, ver = my_box[2], my_box[1]
            if len(image_sequence) == opt.sliding_window_size:  # we have enough frames for prediction, predict for middle frame
                cur_frame = frames[loc]
                cur_bbox = bbox_sequence[loc]
                frames.pop(0)
                bbox_sequence.pop(0)
                if not image_sequence[opt.sliding_window_size // 2][1]:  # if middle image is valid
                    if opt.architecture == "icatcher+":
                        to_predict = {"imgs": torch.tensor(np.array([x[0] for x in image_sequence[0::2]]), dtype=torch.float).squeeze().permute(0, 3, 1, 2).to(opt.device),
                                      "boxs": torch.tensor(np.array(box_sequence[::2]), dtype=torch.float).to(opt.device)
                                      }
                    elif opt.architecture == "icatcher_vanilla":
                        to_predict = {"imgs": torch.tensor([x[0] for x in image_sequence[0::2]], dtype=torch.float).permute(1, 0, 4, 2, 3).to(opt.device)}
                    else:
                        raise NotImplementedError
                    with torch.set_grad_enabled(False):
                        outputs = primary_model(to_predict)
                        probs = torch.nn.functional.softmax(outputs, dim=1)
                        _, prediction = torch.max(outputs, 1)
                        confidence, _ = torch.max(probs, 1)
                        float32_conf = confidence.cpu().numpy()[0]
                        int32_pred = prediction.cpu().numpy()[0]
                    answers[loc] = int32_pred
                    confidences[loc] = float32_conf
                image_sequence.pop(0)
                box_sequence.pop(0)
                class_text = reverse_classes[answers[loc]]
                if opt.on_off:
                    class_text = "off" if class_text == "away" else "on"
                if opt.show_output:
                    prepped_frame = visualize.prep_frame(cur_frame, cur_bbox,
                                                         show_arrow=True, conf=confidences[loc], class_text=class_text)
                    cv2.imshow('frame', prepped_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                if opt.output_video_path:
                    prepped_frame = visualize.prep_frame(cur_frame, cur_bbox,
                                                         show_arrow=True, conf=confidences[loc], class_text=class_text)
                    video_output.write(prepped_frame)
                # handle writing output to file
                if opt.output_annotation:
                    if opt.output_format == "raw_output":
                        output_file.write("{}, {}, {:.02f}\n".format(str(frame_count + loc + 1), class_text, confidences[loc]))
                    elif opt.output_format == "PrefLookTimestamp":
                        if class_text != last_class_text:  # Record "event" for change of direction if code has changed
                            frame_ms = int((frame_count + loc + 1) * (1000. / framerate))
                            output_file.write("{},0,{}\n".format(frame_ms, class_text))
                            last_class_text = class_text
                logging.info("frame: {}, class: {}, confidence: {:.02f}, cur_fps: {:.02f}".format(str(frame_count + loc + 1), class_text, confidences[loc], cur_fps()))
            ret_val, frame = cap.read()
            frame_count += 1
        # finished processing a video file, cleanup
        if opt.show_output:
            cv2.destroyAllWindows()
        if opt.output_video_path:
            video_output.release()
        if opt.output_annotation:  # write footer to file
            if opt.output_format == "PrefLookTimestamp":
                start_ms = int((1000. / framerate) * (opt.sliding_window_size // 2))
                end_ms = int((1000. / framerate) * frame_count)
                output_file.write("{},{},codingactive\n".format(start_ms, end_ms))
                output_file.close()
            elif opt.output_format == "compressed":
                np.savez(my_output_file_path, answers, confidences)
            elif opt.output_format == "raw_output":
                output_file.close()
        cap.release()


if __name__ == '__main__':
    args = options.parse_arguments_for_testing()
    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(filename=args.log, filemode='w', level=args.verbosity.upper())
    else:
        logging.basicConfig(level=args.verbosity.upper())
    predict_from_video(args)
