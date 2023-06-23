import ffmpeg
import subprocess
import re
import sys
from pathlib import Path
import time
import collections
import cv2
import logging

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

def get_video_stream_meta_data(video_file_path):
    probe = ffmpeg.probe(str(video_file_path))
    video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
    return video_info

def get_fps(video_file_path, is_vfr=False):
    """
    get the fps of a video using ffmpeg
    """
    meta_data = get_video_stream_meta_data(video_file_path)
    if is_vfr:
        fps = int(meta_data['avg_frame_rate'].split('/')[0]) / int(meta_data['avg_frame_rate'].split('/')[1])
    else:
        fps = int(meta_data['r_frame_rate'].split('/')[0]) / int(meta_data['r_frame_rate'].split('/')[1])
    return fps

def is_video_vfr(video_file_path, get_meta_data=False):
    """
    checks if video is vfr using ffmpeg
    """
    # these three cases cover windows linux and MacOS hopefully
    ENVBIN = Path(sys.exec_prefix, "bin", "ffmpeg")
    if not ENVBIN.exists():
        ENVBIN = Path("ffmpeg.exe")
    if not ENVBIN.exists():
        ENVBIN = Path("ffmpeg")
    args = [str(ENVBIN)+" ",
            "-i \"{}\"".format(str(video_file_path)),
            "-vf vfrdet",
            "-f null -max_muxing_queue_size 9999 -"]  # 
    p = subprocess.Popen(" ".join(args), stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    out, err = p.communicate()
    if p.returncode != 0:
        print('ffmpeg', out, err)
        exit()
    else:
        output = err.decode('utf-8')
        vfr_str = re.findall("VFR:\d+\.\d+", output)[-1].split(":")[-1]
        vfr = float(vfr_str)
    if get_meta_data:
        meta_data = get_video_stream_meta_data(video_file_path)
        return vfr != 0.0, meta_data
    else:
        return vfr != 0.0
    
def process_video(video_path, opt):
    """
    give a video path, process it and return a generator to iterate over frames
    :param video_path: the video path
    :param opt: command line options
    :return: a generator to iterate over frames, framerate, resolution, and height/width pixel coordinates to crop from
    """
    cap = cv2.VideoCapture(str(video_path))
    # Get some basic info about the video
    vfr, meta_data = is_video_vfr(video_path, get_meta_data=True)
    framerate = get_fps(video_path, vfr)
    if vfr:
        logging.warning("video file: {} has variable frame rate, iCatcher+ underperforms for vfr videos.".format(str(video_path.name)))
        logging.debug("printing video metadata...")
        logging.debug(str(meta_data))
    else:
        logging.debug("video fps: {}".format(framerate))
    raw_width = meta_data["width"]
    raw_height = meta_data["height"]
    resolution = (int(raw_width), int(raw_height))
    cropped_height = raw_height
    if "top" in opt.crop_mode:
        cropped_height = int(raw_height * (1 - (opt.crop_percent / 100)))  # x% of the video from the top
    cropped_width = raw_width
    if "left" in opt.crop_mode and "right" in opt.crop_mode:
        cropped_width = int(raw_width * (1 - (2*opt.crop_percent / 100)))  # x% of the video from both left/right
    elif "left" in opt.crop_mode or "right" in opt.crop_mode:
        cropped_width = int(raw_width * (1 - (opt.crop_percent / 100)))  # x% of the video from both left/right
    h_start_at = (raw_height - cropped_height)
    h_end_at = raw_height
    if "left" in opt.crop_mode and "right" in opt.crop_mode:
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