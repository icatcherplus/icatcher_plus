import json
import cv2
import numpy as np
from pathlib import Path
from icatcher import draw

from typing import Callable, Dict, Union, Tuple


def prepare_ui_output_components(
    ui_packaging_path: str, video_path: str, video_creator: Callable
) -> Dict[str, Union[cv2.VideoWriter, str]]:
    """
    Given a path to a directory, prepares a dictionary of paths and videos necessary for the UI.

    :param ui_packaging_path: path to folder in which the output will be saved
    :param video_path: the original video path
    :param video_creator: a function to create video files given a path
    :return: a dictionary mapping each UI component to its path or video writer
    """

    original_video_path = Path(ui_packaging_path, video_path.stem, "video.mp4")
    decorated_video_path = Path(
        ui_packaging_path, video_path.stem, "decorated_video.mp4"
    )
    bbox_only_video_path = Path(
        ui_packaging_path, video_path.stem, "decorated_video_bbox_only.mp4"
    )

    frames_path = Path(ui_packaging_path, video_path.stem, "frames")
    decorated_frames_path = Path(ui_packaging_path, video_path.stem, "decorated_frames")
    bbox_only_frames_path = Path(
        ui_packaging_path, video_path.stem, "decorated_frames_bbox_only"
    )

    frames_path.mkdir(parents=True, exist_ok=True)
    decorated_frames_path.mkdir(parents=True, exist_ok=True)
    bbox_only_frames_path.mkdir(parents=True, exist_ok=True)

    labels_path = Path(ui_packaging_path, video_path.stem, "labels.txt")
    metadata_path = Path(ui_packaging_path, video_path.stem, "metadata.json")

    ui_output_components = {
        "original_video": video_creator(original_video_path),
        "decorated_video": video_creator(decorated_video_path),
        "bbox_only_video": video_creator(bbox_only_video_path),
        "frames_path": frames_path,
        "decorated_frames_path": decorated_frames_path,
        "bbox_only_frames_path": bbox_only_frames_path,
        "labels_path": labels_path,
        "metadata_path": metadata_path,
    }
    return ui_output_components


def prepare_frame_for_ui(
    cur_frame: np.ndarray,
    cur_bbox: np.ndarray,
    rect_color: Tuple[int, int, int],
    conf: np.ndarray,
    class_text: str,
    frame_number: int,
    pic_in_pic: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """
    Given a frame and decoration parameters, generates variants of the frame without decoration, with bounding boxes only, and
     with full decoration, and generates an annotaiton text to be added to the labels file.

    :param cur_frame: image of the frame to prepare
    :param cur_bbox: bounding box of the face
    :param rect_color: color of the rectangle representing the bounding box
    :param conf: model's prediction confidence
    :param class_text: the predicted class by the model
    :param frame_number: the index of the frame in `cur_frame` in the video
    :param pic_in_pic: whether to show a mini picture with detections
    :return: three images: original image, fully-decorated image, and image with bounding boxes only; and the frame annotation
    """

    decorated_frame = draw.prepare_frame(
        cur_frame.copy(),
        cur_bbox,
        show_arrow=True,
        rect_color=rect_color,
        conf=conf,
        class_text=class_text,
        frame_number=frame_number,
        pic_in_pic=pic_in_pic,
    )

    bbox_only_frame = draw.prepare_frame(
        cur_frame.copy(),
        cur_bbox,
        show_arrow=True,
        rect_color=rect_color,
    )

    label_txt = f"{class_text}, {float(conf):.02}"
    return (
        cur_frame,
        decorated_frame,
        bbox_only_frame,
        label_txt,
    )


def save_ui_output(frame_idx: int, ui_output_components: Dict, output_for_ui: Tuple):
    """
    Given the UI components and inference output, saves the output for the current frame in the UI output directory

    :param frame_idx: number of the current frame to be saved
    :param ui_output_components: dictionary containing UI components and their paths/video writers
    :param output_for_ui: a tuple containing the original frame, decorated frame, frame with bounding boxes only, and
                          annotation text
    """

    original_frame, decorated_frame, bbox_only_frame, label_text = output_for_ui

    # Save raw frame
    ui_output_components["original_video"].write(original_frame)
    original_frame_path = Path(
        ui_output_components["frames_path"], f"frame_{frame_idx:05d}.jpg"
    )
    cv2.imwrite(str(original_frame_path), original_frame)

    # Save decorated frame
    ui_output_components["decorated_video"].write(decorated_frame)
    decorated_frame_path = Path(
        ui_output_components["decorated_frames_path"], f"frame_{frame_idx:05d}.jpg"
    )
    cv2.imwrite(str(decorated_frame_path), decorated_frame)

    # Save decorated frame
    ui_output_components["bbox_only_video"].write(bbox_only_frame)
    bbox_only_frame_path = Path(
        ui_output_components["bbox_only_frames_path"], f"frame_{frame_idx:05d}.jpg"
    )
    cv2.imwrite(str(bbox_only_frame_path), bbox_only_frame)

    # Wrtie new annotation to labels file
    with open(ui_output_components["labels_path"], "a", newline="") as f:
        f.write(f"{frame_idx}, {label_text}\n")


def save_ui_metadata(
    fps: float, frame_count: int, sliding_window_size: int, metadata_file_path: Path
) -> Dict[str, Union[float, int, str]]:
    """
    Given metadata information on the video and annotation process, compiles a dict with
     metadata and saves it as a json file.

    :param fps: frames per second in the saved videos
    :param frame_count: number of frames in the saved videos
    :param sliding_window_size: number of frames in rolling window of each datapoint
    :param metadata_file_path: path of the file to save the metadata to using JSON
    :return: a dictionary contiaining video and inference metadata for UI visualization
    """

    fps = round(fps, 2)
    metadata = {
        "fps": fps,
        "numFrames": frame_count - sliding_window_size + 1,
        "frameOffset": sliding_window_size - 1,
        "metadataExample": {
            "baseFramePath": str(Path(metadata_file_path.parent, "frames/")),
            "baseFileName": "frame",
            "numDigitsFrame": 5,
            "frameExt": ".jpg",
        },
    }

    with open(metadata_file_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return metadata
