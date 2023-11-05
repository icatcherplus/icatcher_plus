import argparse
from pathlib import Path
from . import version
from pathos.helpers import cpu_count


def parse_arguments(my_string=None):
    """
    parse command line arguments
    :param my_string: if provided, will parse this string instead of command line arguments
    :return: parsed arguments
    """
    parser = argparse.ArgumentParser(prog="icatcher")
    parser.add_argument(
        "source",
        type=str,
        help="The source to use (path to video file, folder or webcam id).",
        nargs='?',
        default=None
    )
    parser.add_argument(
        "-a", "--app",
        action="store_true",
        help="Model file that will be used for gaze detection.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="icatcher+_lookit_regnet.pth",
        choices=[
            "icatcher+_lookit.pth",
            "icatcher+_lookit_regnet.pth",
            "icatcher+_bw-cali.pth",
            "icatcher+_senegal.pth",
        ],
        help="Model file that will be used for gaze detection.",
    )
    parser.add_argument(
        "--fd_model",
        type=str,
        choices=["retinaface", "opencv_dnn"],
        default="retinaface",
        help="The face detector model used. opencv_dnn may be more suitable for cpu usage if speed is priority over accuracy.",
    )
    parser.add_argument(
        "--use_fc_model",
        action="store_true",
        help="If supplied, will use face classifier "
        "to decide which crop to use from every frame.",
    )
    parser.add_argument(
        "--fc_model",
        type=str,
        default="face_classifier_lookit.pth",
        choices=[
            "face_classifier_lookit.pth",
            "face_classifier_cali-bw.pth",
            "face_classifier_senegal.pth",
        ],
        help="Face classifier model file that will be used for deciding "
        "which crop should we select from every frame.",
    )
    parser.add_argument(
        "--source_type",
        type=str,
        default="file",
        choices=["file", "webcam"],
        help="Selects source of stream to use.",
    )
    parser.add_argument(
        "--crop_percent",
        type=int,
        default=0,
        help="A percent to crop video frames to prevent other people from appearing.",
    )
    parser.add_argument(
        "--crop_mode",
        type=str,
        choices=["top", "left", "right"],
        nargs="+",
        default=["top"],
        help="Where to crop video from, multi-choice.",
    )
    parser.add_argument(
        "--show_output",
        action="store_true",
        help="Show results online in a separate window.",
    )
    parser.add_argument(
        "--output_annotation", type=str, help="Folder to output annotations to."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If an output annotation file exists, will overwrite it. Without this flag iCatcher+ will terminate upon encountering an existing annotation file.",
    )
    parser.add_argument(
        "--on_off",
        action="store_true",
        help="Left/right/away annotations will be swapped with on/off.",
    )
    parser.add_argument(
        "--mirror_annotation",
        action="store_true",
        help="Left will be swapped with right, and right will be swapped with left.",
    )
    parser.add_argument(
        "--output_format",
        type=str,
        default="raw_output",
        choices=["raw_output", "compressed"],
    )
    parser.add_argument(
        "--output_video_path",
        help="If present, annotated video will be saved to this folder.",
    )
    parser.add_argument(
        "--ui_packaging_path",
        help="If present, packages the output data into the UI format.",
    )
    parser.add_argument(
        "--pic_in_pic",
        action="store_true",
        help="If present, a mini picture with detections will be shown in the output video.",
    )
    parser.add_argument(
        "--output_file_suffix", type=str, default=".txt", help="The output file suffix."
    )
    parser.add_argument(
        "--per_channel_mean",
        nargs=3,
        metavar=("Channel1_mean", "Channel2_mean", "Channel3_mean"),
        type=float,
        default=[0.485, 0.456, 0.406],
        help="Supply custom per-channel mean of data for normalization.",
    )
    parser.add_argument(
        "--per_channel_std",
        nargs=3,
        metavar=("Channel1_std", "Channel2_std", "Channel3_std"),
        type=float,
        default=[0.229, 0.224, 0.225],
        help="Supply custom per-channel std of data for normalization.",
    )
    parser.add_argument(
        "--gpu_id", type=int, default=-1, help="GPU id to use, use -1 for CPU."
    )
    parser.add_argument("--log", help="If present, writes log to this path")
    parser.add_argument(
        "--verbosity",
        type=str,
        choices=["debug", "info", "warning"],
        default="info",
        help="Selects verbosity level.",
    )
    parser.add_argument(
        "--video_filter",
        type=str,
        help="Provided file will be used to filter only test videos,"
        " will assume certain file structure using the lookit/cali-bw/senegal datasets.",
    )
    parser.add_argument(
        "--illegal_transitions_path",
        type=str,
        help="Path to CSV with illegal transitions to 'smooth' over.",
    )
    parser.add_argument("--version", action="version", version="%(prog)s " + version)
    # face detection options:
    parser.add_argument(
        "--fd_confidence_threshold",
        type=float,
        help="The score confidence threshold that needs to be met for a face to be detected.",
    )
    parser.add_argument(
        "--fd_parallel_processing",
        action="store_true",
        default=False,
        help="(cpu, retinaface only) face detection will be parallelized, by batching the frames (requires buffering them), increasing memory usage, but decreasing overall processing time. Disallows live stream of results.",
    )
    parser.add_argument(
        "--fd_num_cpus",
        type=int,
        default=-1,
        help="(cpu, retinaface only) amount of cpus to use if face detection parallel processing is true (-1: use all available cpus)).",
    )
    parser.add_argument(
        "--fd_batch_size",
        type=int,
        default=16,
        help="(cpu, retinaface only) amount of frames fed at once into face detector if parallel processing is true.",
    )
    parser.add_argument(
        "--fd_skip_frames",
        type=int,
        default=0,
        help="(cpu, retinaface only) amount of frames to skip between each face detection if parallel processing is true. previous bbox will be used.",
    )
    parser.add_argument(
        "--track_face",
        action="store_true",
        help="If detection is lost, will keep track of face using last known position. WARNING: untested experimental feature.",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=100,
        help="All images will be resized to this size. WARNING: changing default results in untested behavior.",
    )
    parser.add_argument(
        "--sliding_window_size",
        type=int,
        default=9,
        help="Number of frames in rolling window of each datapoint. WARNING: changing default results in untested behavior.",
    )
    parser.add_argument(
        "--window_stride",
        type=int,
        default=2,
        help="Stride between frames in rolling window. WARNING: changing default results in untested behavior.",
    )
    if my_string is not None:
        args = parser.parse_args(my_string.split())
    else:
        args = parser.parse_args()
    if (
        args.fd_confidence_threshold is None
    ):  # set defaults outside argparse to avoid complication
        if args.fd_model == "retinaface":
            args.fd_confidence_threshold = 0.9
        elif args.fd_model == "opencv_dnn":
            args.fd_confidence_threshold = 0.7
    if args.crop_percent not in [x for x in range(100)]:
        raise ValueError("crop_video must be a percent between 0 - 99")
    if "left" in args.crop_mode and "right" in args.crop_mode:
        if args.crop_percent > 49:
            raise ValueError(
                "crop_video must be a percent between 0 - 49 when cropping both sides"
            )
    if args.video_filter:
        args.video_filter = Path(args.video_filter)
        if not args.video_filter.is_file() and not args.video_filter.is_dir():
            raise FileNotFoundError("Video filter is not a file or a folder")
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
            raise ValueError(
                "On off mode can only be used with raw output format. Pass raw_output with the --output_format flag."
            )
    if args.sliding_window_size % 2 == 0:
        raise ValueError("sliding_window_size must be odd.")
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
    # figure out how many cpus can be used
    use_cpu = True if args.gpu_id == -1 else False
    if use_cpu:
        if args.fd_num_cpus == -1:
            args.fd_num_cpus = cpu_count()
        else:
            if args.fd_num_cpus > cpu_count():
                raise ValueError(
                    "Number of cpus requested is greater than available cpus"
                )
    return args
