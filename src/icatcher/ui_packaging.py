import cv2
import numpy as np
from pathlib import Path
from icatcher import draw
from typing import Dict, Tuple


def prepare_ui_output_components(
    ui_packaging_path: str, video_path: str, overwrite: bool
) -> Dict[str, str]:
    """
    Given a path to a directory, prepares a dictionary of paths and videos necessary for the UI.

    :param ui_packaging_path: path to folder in which the output will be saved
    :param video_path: the original video path
    :param overwrite: if true and label file already exists, overwrites it. else will throw an error.
    :return: a dictionary mapping each UI component to its path or video writer
    """

    labels_path = Path(ui_packaging_path, video_path.stem, "labels.txt")
    if labels_path.exists():
        if overwrite:
            labels_path.unlink()
        else:
            raise FileExistsError(
                "Annotation output file already exists. Use --overwrite flag to overwrite."
            )
    decorated_frames_path = Path(ui_packaging_path, video_path.stem, "decorated_frames")
    decorated_frames_path.mkdir(parents=True, exist_ok=True)
    ui_output_components = {
        "decorated_frames_path": decorated_frames_path,
        "labels_path": labels_path,
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
) -> Tuple[np.ndarray, str]:
    """
    Given a frame and decoration parameters, generates frame with full decoration,
     and an annotation text to be added to the labels file.

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

    label_txt = f"{class_text}, {float(conf):.02}"
    return (
        decorated_frame,
        label_txt,
    )


def save_ui_output(frame_idx: int, ui_output_components: Dict, output_for_ui: Tuple):
    """
    Given the UI components and inference output, saves the output for the current frame in the UI output directory

    :param frame_idx: number of the current frame to be saved
    :param ui_output_components: dictionary containing UI components and their paths/video writers
    :param output_for_ui: a tuple containing the decorated frame and annotation text
    """

    decorated_frame, label_text = output_for_ui

    # Save decorated frame
    decorated_frame_path = Path(
        ui_output_components["decorated_frames_path"], f"frame_{frame_idx:05d}.jpg"
    )
    cv2.imwrite(str(decorated_frame_path), decorated_frame)
    # Write new annotation to labels file
    with open(ui_output_components["labels_path"], "a", newline="") as f:
        f.write(f"{frame_idx}, {label_text}\n")
