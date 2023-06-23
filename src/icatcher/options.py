import argparse
from pathlib import Path
from . import version


def parse_arguments(my_string=None):
    """
    parse command line arguments
    :param my_string: if provided, will parse this string instead of command line arguments
    :return: parsed arguments
    """
    parser = argparse.ArgumentParser(prog='icatcher')
    parser.add_argument("source", type=str, help="the source to use (path to video file, folder or webcam id)")
    parser.add_argument("--model", type=str, help="path to model that will be used for predictions "
                                                  "if not supplied will use model trained on the lookit dataset")
    parser.add_argument("--use_fc_model", action="store_true", help="if supplied, will use face classifier "
                                                                              "to decide which crop to use from every frame.")
    parser.add_argument("--fc_model", type=str, help="path to face classifier model that will be used for deciding "
                                                     "which crop should we select from every frame. "
                                                     "if not supplied but use_fc_model is true, will use the model trained on the lookit dataset.")
    parser.add_argument("--source_type", type=str, default="file", choices=["file", "webcam"],
                        help="selects source of stream to use.")
    parser.add_argument("--crop_percent", type=int, default=0, help="A percent to crop video frames to prevent other people from appearing")
    parser.add_argument("--crop_mode", type=str, choices=["top", "left", "right"], nargs="+", default=["top"], help="where to crop video from, multi-choice.")
    parser.add_argument("--track_face", action="store_true", help="if detection is lost, will keep track of face using last known position.")
    parser.add_argument("--show_output", action="store_true", help="show results online in a separate window")
    parser.add_argument("--output_annotation", type=str, help="folder to output annotations to")
    parser.add_argument("--on_off", action="store_true",
                        help="left/right/away annotations will be swapped with on/off (only works with icatcher+)")
    parser.add_argument("--output_format", type=str, default="raw_output", choices=["raw_output",
                                                                                    "compressed",
                                                                                    "PrefLookTimestamp"])  # https://osf.io/3n97m/ - PrefLookTimestamp coding standard
    parser.add_argument("--output_video_path", help="if present, annotated video will be saved to this folder")
    parser.add_argument("--pic_in_pic", action="store_true", help="if present, a mini picture with detection will be shown in the output video")
    parser.add_argument("--output_file_suffix", type=str, default=".txt", help="the output file suffix")
    parser.add_argument("--image_size", type=int, default=100, help="All images will be resized to this size")
    parser.add_argument("--sliding_window_size", type=int, default=9, help="Number of frames in rolling window of each datapoint")
    parser.add_argument("--window_stride", type=int, default=2, help="Stride between frames in rolling window")
    parser.add_argument('--per_channel_mean', nargs=3, metavar=('Channel1_mean', 'Channel2_mean', 'Channel3_mean'),
                        type=float, default=[0.485, 0.456, 0.406],
                        help='supply custom per-channel mean of data for normalization')
    parser.add_argument('--per_channel_std', nargs=3, metavar=('Channel1_std', 'Channel2_std', 'Channel3_std'),
                        type=float, default=[0.229, 0.224, 0.225],
                        help='supply custom per-channel std of data for normalization')
    parser.add_argument("--gpu_id", type=int, default=-1, help="GPU id to use, use -1 for CPU.")
    parser.add_argument("--log",
                        help="If present, writes log to this path")
    parser.add_argument("-v", "--verbosity", type=str, choices=["debug", "info", "warning"], default="info",
                        help="Selects verbosity level")
    parser.add_argument("--video_filter", type=str,
                        help="provided file will be used to filter only test videos,"
                             " will assume certain file structure using the lookit/cali-bw/senegal datasets")
    parser.add_argument("--raw_dataset_path", type=str, help="path to raw dataset (required if --video_filter is passed")
    parser.add_argument("--raw_dataset_type", type=str, choices=["lookit", "cali-bw", "senegal", "generic"], default="lookit",
                        help="the type of dataset to preprocess")
    parser.add_argument("--illegal_transitions_path", type=str, help="path to CSV with illegal transitions to 'smooth' over")
    parser.add_argument('--version', action='version', version="%(prog)s "+version)
    # face detection options:
    parser.add_argument("--fd_model", type=str, choices=["retinaface", "opencv_dnn"], default="retinaface",
                        help="the face detector model used. opencv_dnn may be more suitable for cpu usage if speed is priority over accuracy")
    parser.add_argument("--fd_confidence_threshold", type=float, help="the score confidence threshold that needs to be met for a face to be detected")
    parser.add_argument("--num_cpus_saved", type=int, default=0,
                        help="(retinaface only) amount of cpus to not use in parallel processing of face detection")
    parser.add_argument("--fd_batch_size", type=int, default=16,
                        help="(retinaface only) amount of frames fed into face detector at one time for batch inference")
    parser.add_argument("--fd_skip_frames", type=int, default=0,
                        help="(cpu only) amount of frames to skip between each face detection. previous bbox will be used")
    parser.add_argument("--dont_buffer", action="store_true", default=False,
                        help="(cpu, retinaface only) frames will not be buffered, decreasing memory usage, but increasing processing time. Allows live stream of results.")
    if my_string is not None:
        args = parser.parse_args(my_string.split())
    else:
        args = parser.parse_args()
    if args.model:
        args.model = Path(args.model)
    if args.fd_confidence_threshold is None:  # set defaults outside argparse to avoid complication
        if args.fd_model == "retinaface":
            args.fd_confidence_threshold = 0.9
        elif args.fd_model == "opencv_dnn":
            args.fd_confidence_threshold = 0.7
    # if not args.model.is_file():
    #     raise FileNotFoundError("Model file not found")
    if args.crop_percent not in [x for x in range(100)]:
        raise ValueError("crop_video must be a percent between 0 - 99")
    if "left" in args.crop_mode and "right" in args.crop_mode:
        if args.crop_percent > 49:
            raise ValueError("crop_video must be a percent between 0 - 49 when cropping both sides")
    if args.video_filter:
        args.video_filter = Path(args.video_filter)
        if not args.video_filter.is_file() and not args.video_filter.is_dir():
            raise FileNotFoundError("Video filter is not a file or a folder")
    if args.raw_dataset_path:
        args.raw_dataset_path = Path(args.raw_dataset_path)
    if args.output_annotation:
        args.output_annotation = Path(args.output_annotation)
        args.output_annotation.mkdir(exist_ok=True, parents=True)
    if args.output_video_path:
        args.output_video_path = Path(args.output_video_path)
        args.output_video_path.mkdir(exist_ok=True, parents=True)
    if args.log:
        args.log = Path(args.log)
    if args.on_off:
        if args.output_format != "raw_output":
            raise ValueError("On off mode can only be used with raw output format. Pass raw_output with the --output_format flag.")
    if not args.per_channel_mean:
        args.per_channel_mean = [0.485, 0.456, 0.406]
    if not args.per_channel_std:
        args.per_channel_std = [0.229, 0.224, 0.225]
    if args.gpu_id == -1:
        args.device = "cpu"
    else:
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        args.device = "cuda:{}".format(0)
        import torch
        if not torch.cuda.is_available():
            raise ValueError("GPU is not available. Was torch compiled with CUDA?")
    return args
