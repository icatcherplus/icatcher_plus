import cv2
import numpy as np


def put_text(img, text, loc=None):
    """
    inserts a text into image
    :param img:
    :param class_name:
    :param loc:
    :return:
    """
    font = cv2.FONT_HERSHEY_DUPLEX
    if loc is not None:
        text_location = loc
    else:
        text_location = (10, 30)  # top_left_corner_text
    font_scale = 1
    font_color = (255, 255, 255)
    bg_color = (0,0,0)
    line_type = 2
    
    text_size, _ = cv2.getTextSize(text, font, font_scale, line_type)
    text_w, text_h = text_size
    
    cv2.rectangle(img, text_location, (text_location[0] + text_w, text_location[1] + text_h), bg_color, -1)

    cv2.putText(img, text, (text_location[0], text_location[1] + text_h + font_scale - 1),
                font,
                font_scale,
                font_color,
                line_type)
    
    return img

def put_arrow(img, class_name, face):
    """
    inserts an arrow into a frame
    :param img: the frame
    :param class_name: this will dictate where the arrow will face
    :param face: bounding box of face in frame
    :return: the frame with an arrow
    """
    arrow_start_x = int(face[0] + 0.5 * face[2])
    arrow_end_x = int(face[0] + 0.1 * face[2] if class_name == "left" else face[0] + 0.9 * face[2])
    arrow_y = int(face[1] + 0.8 * face[3])
    img = cv2.arrowedLine(img,
                          (arrow_start_x, arrow_y),
                          (arrow_end_x, arrow_y),
                          (0, 255, 0),
                          thickness=3,
                          tipLength=0.4)
    return img

def put_rectangle(frame, rec, color=None):
    """
    inserts a rectangle in frame
    :param frame: the frame
    :param rec: the bounding box of the rectangle
    :return:
    """
    if color is None:
        color = (0, 255, 0)  # green
    thickness = 2
    frame = cv2.rectangle(frame,
                          (rec[0], rec[1]), (rec[0] + rec[2], rec[1] + rec[3]),
                          color,
                          thickness)
    return frame

def prepare_frame(frame, bbox, show_bbox=True, show_arrow=False, conf=None, class_text=None, rect_color=None, frame_number=None, pic_in_pic=False):
    """
    prepares a frame for visualization by adding text, rectangles and arrows.
    :param frame: the frame for which to add the gizmo's to
    :param bbox: bbox as in cv2
    :param show_bbox: to show bbox on face as a green rectangle
    :param show_arrow: to show arrow indicating direciton of looking
    :param conf: confidence of the classifier
    :param class_text: class text to show on top left corner
    :param rect_color: color of the rectangle
    :param frame_number: frame number to show on top left corner
    :param pic_in_pic: to show a small frame of the face
    :return:
    """
    if show_arrow:
        if class_text is not None and bbox is not None:
            if class_text == "right" or class_text == "left":
                frame = put_arrow(frame, class_text, bbox)
    if conf and bbox is not None:
        frame = put_text(frame, "{:.02f}".format(conf),
                         loc=(bbox[0], bbox[1] + bbox[3]))
    if pic_in_pic:
        pic_in_pic_size = 100
        if bbox is None:
            crop_img = np.zeros((pic_in_pic_size, pic_in_pic_size, 3), np.uint8)
        else:
            crop_img = frame[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]
            crop_img = cv2.resize(crop_img, (pic_in_pic_size, pic_in_pic_size))
        frame[frame.shape[0]-pic_in_pic_size:, :pic_in_pic_size] = crop_img
    if class_text is not None:
        frame = put_text(frame, class_text)
    if show_bbox and bbox is not None:
        frame = put_rectangle(frame, bbox, rect_color)
    if frame_number is not None:  # may fail if loc outside resolution
        frame = put_text(frame, str(frame_number), loc=(10,70))
    return frame

def mask_regions(image, start_h, end_h, start_w, end_w):
    """
    masks a numpy image with black background outside of region of interest (roi)
    :param image: numpy image h x w x c
    :param start_h: where does the roi height start
    :param end_h: where does the roi height end
    :param start_w: where does the roi width start
    :param end_w: where does the roi width end
    :return: masked image
    """
    h, w, _ = image.shape
    if start_h < 0 or start_w < 0 or end_h > h or end_w > w:
        raise ValueError("Values exceed image resolution")
    output = np.zeros_like(image)
    output[start_h:end_h, start_w:end_w, :] = image[start_h:end_h, start_w:end_w, :]
    return output
