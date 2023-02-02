import ffmpeg
import subprocess
import re
import sys
from pathlib import Path
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

def get_video_stream_meta_data(video_file_path):
    probe = ffmpeg.probe(str(video_file_path))
    video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
    return video_info


def get_fps(video_file_path):
    meta_data = get_video_stream_meta_data(video_file_path)
    return int(meta_data['r_frame_rate'].split('/')[0]) / int(meta_data['r_frame_rate'].split('/')[1])


def is_video_vfr(video_file_path, get_meta_data=False):
    """
    checks if video is vfr
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
            "-f null -"]
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