import ffmpeg
import logging
import subprocess
import re
import sys
from pathlib import Path


def get_video_stream_meta_data(video_file_path):
    probe = ffmpeg.probe(str(video_file_path))
    video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
    return video_info


def get_fps(video_file_path):
    meta_data = get_video_stream_meta_data(video_file_path)
    return int(meta_data['r_frame_rate'].split('/')[0]) / int(meta_data['r_frame_rate'].split('/')[1])


def is_video_vfr(video_file_path, get_meta_data=False):
    ENVBIN = Path(sys.exec_prefix, "bin", "ffmpeg")
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


def get_frame_information(video_file_path):
    output = ffmpeg.probe(str(video_file_path), show_frames="-show_frames")
    video_frames = [frame for frame in output['frames'] if frame['media_type'] == 'video']
    frame_times = [frame["pkt_pts_time"] for frame in video_frames]
    video_stream_info = next(s for s in output['streams'] if s['codec_type'] == 'video')
    assert len(frame_times) == int(video_stream_info["nb_frames"])
    frame_times_ms = [1000*float(x) for x in frame_times]
    assert frame_times_ms[0] == 0.0
    # returns timestamps in milliseconds
    return frame_times_ms, int(video_stream_info["nb_frames"]), video_stream_info["time_base"]
