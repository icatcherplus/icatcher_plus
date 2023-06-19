import os
from pathlib import Path
import pickle
import itertools
import logging
from tqdm import tqdm
import numpy as np
import cv2
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, cohen_kappa_score
import pingouin as pg
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
from matplotlib.patches import Patch
from options import parse_arguments_for_visualizations
import parsers
import preprocess
from statistics.bootstrap import bootstrap, t_test, t_test_paired
import warnings


def label_to_color(label):
    mapping = {"left": (0.031, 0.411, 0.643),
               "right": (0.823, 0.309, 0),
               "away": "lightgrey",
               "invalid": "white",
               "lblue": (0.5, 0.6, 0.9),
               "lred": (0.9, 0.6, 0.5),
               "lgreen": (0.6, 0.8, 0.0),
               "lorange": (0.94, 0.78, 0.0),
               "lyellow": (0.9, 0.9, 0.0),
               "mblue": (0.12, 0.41, 0.87),
               "cblind_red": (0.823, 0.309, 0),
               "cblind_blue": (0.031, 0.411, 0.643),
               "vlblue": (0.086, 0.568, 0.874),
               "vblue": (0.074, 0.349, 0.525),
               "vlgreen": (0.345, 0.890, 0.270),
               "vgreen": (0.149, 0.615, 0.082),
               "vlpurple": (167/255, 118/255, 181/255),
               "vpurple": (157/255, 42/255, 189/255)}
    return mapping[label]


def calculate_confusion_matrix(label, pred, save_path=None, mat=None, class_num=3, flip_xy=False, verbose=True):
    """
    creates a plot of the confusion matrix given the gt labels abd the predictions.
    if mat is supplied, ignores other inputs and uses that.
    :param label: the labels (will be y axis)
    :param pred: the predicitions (will be x axis)
    :param save_path: path to save plot
    :param mat: a numpy 3x3 array representing the confusion matrix
    :param class_num: number of classes
    :return:
    """
    if class_num == 2:
        class_labels = ['on', 'off']
    elif class_num == 3:
        class_labels = ['away', 'left', 'right']
    else:
        raise ValueError
    if mat is None:
        mat = np.zeros([class_num, class_num])
        pred = np.array(pred)
        label = np.array(label)
        if verbose:
            logging.info('# datapoint: {}'.format(len(label)))
        for i in range(class_num):
            for j in range(class_num):
                mat[i][j] = sum((label == i) & (pred == j))
        if flip_xy:
            mat = mat.T
    if np.all(np.sum(mat, -1, keepdims=True) != 0):
        total_acc = (mat.diagonal().sum() / mat.sum()) * 100
        norm_mat = mat / np.sum(mat, -1, keepdims=True)
    else:
        total_acc = 0
        norm_mat = mat
    if save_path:
        fig, ax = plt.subplots(figsize=(3, 3))
        ax = sns.heatmap(norm_mat, ax=ax, vmin=0, vmax=1, annot=True, fmt='.2%', cbar=False, cmap='Blues')
        ax.set_xticklabels(class_labels)
        ax.set_yticklabels(class_labels)
        plt.axis('equal')
        plt.tight_layout(pad=0.1)
        plt.savefig(save_path)
    if verbose:
        logging.info('acc:{:.4f}%'.format(total_acc))
        logging.info('confusion matrix: {}'.format(mat))
        logging.info('normalized confusion matrix: {}'.format(norm_mat))
    return norm_mat, mat, total_acc


def confusion_mat(targets, preds, classes, normalize=False, plot=False, title="Confusion Matrix", cmap=plt.cm.Blues):
    cm = confusion_matrix(targets, preds)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    if plot:
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig(title + ".png")
        plt.cla()
        plt.clf()

    return cm


def plot_learning_curve(train_perfs, val_perfs, save_dir, isLoss=False):
    epochs = np.arange(1, len(train_perfs) + 1)
    plt.plot(epochs, train_perfs, label="Training set")
    plt.plot(epochs, val_perfs, label="Validation set")
    plt.xlabel("Epochs")
    metric_name = "Loss" if isLoss else "Accuracy"
    plt.ylabel(metric_name)
    plt.title(metric_name, fontsize=16, y=1.002)
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'learning_curve_%s.png' % metric_name))
    plt.cla()
    plt.clf()


def get_stats_in_interval(start, end, coding1, coding2, confidence, valid_class_num, video_id=None, face_folder=None):
    """
    given two codings (single dimensional numpy arrays) and a start and end time,
    calculates various metrics we care about. assumes coding1[i], coding2[i] refer to same time
    :param start: start time of interval
    :param end: end time of interval
    :param coding1: np array (1 dimensional)
    :param coding2: np array (1 dimensional)
    :param confidence: optional np array (1 dimensional) of confidence scores for one of the coding files
    :param valid_class_num: number of valid classes in dataset
    :param video_id: if provided (with face_folder), face statistics of this video id will be extracted too
    :param face_folder: if provided (with video_id), face statistics from this face_folder will be extracted too
    note: face_folder expected to have structure of pre-processed dataset, with a valid face_labels_fc.npy file.
    :return: all metrics in interval
    """
    # fix in case one coding ends sooner than end
    assert start >= 0
    diff1 = end - len(coding1)
    diff2 = end - len(coding2)
    if diff1 > 0:
        coding1 = np.concatenate((coding1, -3*np.ones(diff1, dtype=int)))
    if diff2 > 0:
        coding2 = np.concatenate((coding2, -3*np.ones(diff2, dtype=int)))
        if confidence is not None:
            confidence = np.concatenate((confidence, -1*np.ones(diff2, dtype=int)))
    coding1_interval = coding1[start:end]
    coding2_interval = coding2[start:end]
    mutually_valid_frames = np.logical_and(coding1_interval >= 0, coding2_interval >= 0)
    face_stats = [None, None, None]
    if (video_id is not None) and (face_folder is not None):
        face_stats = get_face_stats(video_id, face_folder, start, end, mutually_valid_frames)

    coding1_interval_mut_valid = coding1_interval[mutually_valid_frames]
    coding2_interval_mut_valid = coding2_interval[mutually_valid_frames]

    coding1_away = np.where(coding1_interval == 0)[0]
    coding2_away = np.where(coding2_interval == 0)[0]
    coding1_left = np.where(coding1_interval == 1)[0]
    coding2_left = np.where(coding2_interval == 1)[0]
    coding1_right = np.where(coding1_interval == 2)[0]
    coding2_right = np.where(coding2_interval == 2)[0]
    coding1_invalid = np.where(coding1_interval < 0)[0]
    coding2_invalid = np.where(coding2_interval < 0)[0]


    n_transitions_1 = np.count_nonzero(np.diff(coding1_interval[mutually_valid_frames]))
    n_transitions_2 = np.count_nonzero(np.diff(coding2_interval[mutually_valid_frames]))

    on_screen_1_sum = np.sum(coding1_interval_mut_valid == 1) + np.sum(coding1_interval_mut_valid == 2)
    on_screen_2_sum = np.sum(coding2_interval_mut_valid == 1) + np.sum(coding2_interval_mut_valid == 2)
    off_screen_1_sum = np.sum(coding1_interval_mut_valid == 0)
    off_screen_2_sum = np.sum(coding2_interval_mut_valid == 0)

    if on_screen_1_sum == 0:
        percent_r_1 = np.nan
    else:
        percent_r_1 = np.sum(coding1_interval_mut_valid == 2) / on_screen_1_sum
    if on_screen_2_sum == 0:
        percent_r_2 = np.nan
    else:
        percent_r_2 = np.sum(coding2_interval_mut_valid == 2) / on_screen_2_sum

    looking_time_1 = on_screen_1_sum
    looking_time_2 = on_screen_2_sum

    equal = coding1_interval == coding2_interval

    equal_and_non_equal = np.sum(equal[mutually_valid_frames]) + np.sum(np.logical_not(equal[mutually_valid_frames]))
    if equal_and_non_equal == 0:
        agreement = np.nan
    else:
        agreement = np.sum(equal[mutually_valid_frames]) / equal_and_non_equal
    if valid_class_num == 3:
        _, mat3, _ = calculate_confusion_matrix(coding1_interval_mut_valid, coding2_interval_mut_valid,
                                                class_num=3, flip_xy=True, verbose=False)
    else:
        mat3 = None

    on_screen_1 = coding1_interval_mut_valid.copy()
    on_screen_1[on_screen_1 > 0] = 1
    on_screen_2 = coding2_interval_mut_valid.copy()
    on_screen_2[on_screen_2 > 0] = 1
    _, mat2, _ = calculate_confusion_matrix(on_screen_1, on_screen_2,
                                            class_num=2, flip_xy=True, verbose=False)
    times_coding1 = {"away": coding1_away,
                     "left": coding1_left,
                     "right": coding1_right,
                     "invalid": coding1_invalid}
    times_coding2 = {"away": coding2_away,
                     "left": coding2_left,
                     "right": coding2_right,
                     "invalid": coding2_invalid}
    raw_coding1 = coding1_interval
    raw_coding2 = coding2_interval
    # ignore warning for true divide (kappa is nan for completely equal codings)
    # ignore warning for empty slice mean (we treat nans later)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        kappa = cohen_kappa_score(coding1_interval_mut_valid, coding2_interval_mut_valid)
        if confidence is not None:
            raw_confidence = confidence[start:end]
            valid_confidence = raw_confidence[mutually_valid_frames]
            equal_confidence = raw_confidence[equal & mutually_valid_frames]
            non_equal_confidence = raw_confidence[~equal & mutually_valid_frames]
            confidence_metrics = [np.mean(equal_confidence), np.mean(non_equal_confidence)]
        else:
            valid_confidence = None
            confidence_metrics = None
    # try:
    #     df = pd.DataFrame({"coder1": coding1_interval_mut_valid,
    #                        "coder2": coding2_interval_mut_valid})
    #     ca = pg.cronbach_alpha(data=df)[0]
    # except AssertionError:
    #     ca = np.nan

    return {"n_frames_in_interval": end - start,
            "mutual_valid_frame_count": np.sum(mutually_valid_frames),
            "raw_coding1": raw_coding1,
            "raw_coding2": raw_coding2,
            "raw_confidence": valid_confidence,
            "confidence_metrics": confidence_metrics,
            "valid_frames_1": np.sum(coding1_interval >= 0),
            "valid_frames_2": np.sum(coding2_interval >= 0),
            "n_transitions_1": n_transitions_1,
            "n_transitions_2": n_transitions_2,
            "percent_r_1": percent_r_1,
            "percent_r_2": percent_r_2,
            "looking_time_1": looking_time_1,
            "looking_time_2": looking_time_2,
            "agreement": agreement,
            "kappa": kappa,
            "confusion_matrix": mat3,
            "confusion_matrix2": mat2,
            "start": start,
            "end": end,
            "times_coding1": times_coding1,
            "times_coding2": times_coding2,
            "label_count_1": [np.sum(coding1_away), np.sum(coding1_left), np.sum(coding1_right), np.sum(coding1_invalid)],
            "label_count_2": [np.sum(coding2_away), np.sum(coding2_left), np.sum(coding2_right), np.sum(coding2_invalid)],
            "avg_face_pixel_density": face_stats[0],
            "avg_face_loc": face_stats[1],
            "avg_face_loc_std": face_stats[2]
            }


def compare_uncollapsed_coding_files(coding1, coding2, intervals, confidence=None, valid_class_num=3, video_id=None, face_folder=None):
    """
    computes various metrics between two codings on a set of intervals
    :param coding1: first coding, uncollapsed numpyarray of events
    :param coding2: second coding, uncollapsed numpyarray of events
    :param intervals: list of lists where each internal list contains 2 entries indicating start and end time of interval.
    :param confidence: confidence of one of the codings, uncollapsed float np array of confidence (-1 for invalid)
    :param valid_class_num: number of valid classes in dataset
    :param video_id: if provided (with face_folder), face statistics of this video id will be extracted too
    :param face_folder: if provided (with video_id), face statistics from this face_folder will be extracted too
    note: intervals are considered as [) i.e. includes start time, but excludes end time.
    :return: array of dictionaries containing various metrics (1 dict per interval)
    """
    results = []
    for i, interval in enumerate(intervals):
        # logging.info("trial: {} / {}".format(i, len(intervals)))
        t_start, t_end = interval[0], interval[1]
        results.append(get_stats_in_interval(t_start, t_end, coding1, coding2, confidence, valid_class_num, video_id, face_folder))
    if len(results) == 1:
        results = results[0]
    return results


def compare_coding_files(human_coding_file, human_coding_file2, machine_coding_file, args):
    """
    compares human coders and machine annotations
    :param human_coding_file:
    :param human_coding_file2:
    :param machine_coding_file:
    :param args:
    :return:
    """
    metrics = {}
    if args.machine_coding_format == "compressed":
        parser = parsers.CompressedParser()
    else:
        raise NotImplementedError
    machine, mstart, mend = parser.parse(machine_coding_file.stem, machine_coding_file)
    machine_confidence = parser.get_confidence(machine_coding_file)
    if args.raw_dataset_type == "datavyu":
        machine[machine > 0] = 1
        valid_class_num = 2
    else:
        valid_class_num = 3
    if args.human_coding_format == "vcx":
        parser = parsers.VCXParser(30, args.raw_dataset_path, args.raw_dataset_type)
    elif args.human_coding_format == "lookit":
        parser = parsers.LookitParser(30)
        postures, _, _ = parser.parse(human_coding_file.stem, human_coding_file, extract_poses=True)
    elif args.human_coding_format == "datavyu":
        parser = parsers.DatavyuParser()
    else:
        raise NotImplementedError
    human, start1, end1 = parser.parse(human_coding_file.stem, human_coding_file)
    if human_coding_file2:
        human2, start2, end2 = parser.parse(human_coding_file2.stem, human_coding_file2)
        if end1 != end2:
            logging.warning("humans don't agree on ending: {}".format(human_coding_file))
            logging.warning("frame diff: {}".format(np.abs(end1 - end2)))
            logging.warning("using human1 reported end.")
    if args.human_coding_format == "lookit":
        labels = parser.load_and_sort(human_coding_file)
        trial_times = parser.get_trial_intervals(start1, labels)
        posture_class_map = {x: i for i, x in enumerate(parser.poses)}
        uncollapsed_postures = parser.uncollapse_labels(postures, start1, end1, class_map=posture_class_map)
        metrics["postures"] = uncollapsed_postures
    elif args.human_coding_format == "vcx":
        trial_times = parser.get_trial_intervals(start1, human)
    elif args.human_coding_format == "datavyu":
        trial_times = parser.get_trial_intervals(start1, human_coding_file)
    else:
        raise NotImplementedError
    machine_uncol = parser.uncollapse_labels(machine, mstart, mend)
    # create a version of machine annotation where machine "invalid" is replaced with "away"
    special_machine = machine_uncol.copy()
    special_machine[special_machine < 0] = 0
    human1_uncol = parser.uncollapse_labels(human, start1, end1)
    logging.info("trial level stats")
    metrics["human1_vs_machine_trials"] = compare_uncollapsed_coding_files(human1_uncol, machine_uncol, trial_times,
                                                                           confidence=machine_confidence,
                                                                           valid_class_num=valid_class_num,
                                                                           video_id=human_coding_file.stem,
                                                                           face_folder=args.faces_folder)
    if human_coding_file2:
        human2_uncol = parser.uncollapse_labels(human2, start2, end2)
        metrics["human1_vs_human2_trials"] = compare_uncollapsed_coding_files(human1_uncol, human2_uncol, trial_times)
        ICC_looking_time_hvh = calc_ICC(metrics["human1_vs_human2_trials"],
                                        "looking_time_1", "looking_time_2")
        ICC_percent_r_hvh = calc_ICC(metrics["human1_vs_human2_trials"],
                                     "percent_r_1", "percent_r_2")
    else:
        ICC_looking_time_hvh = None
        ICC_percent_r_hvh = None
    ICC_looking_time_hvm = calc_ICC(metrics["human1_vs_machine_trials"],
                                    "looking_time_1", "looking_time_2")
    if not args.raw_dataset_type == "datavyu":
        ICC_percent_r_hvm = calc_ICC(metrics["human1_vs_machine_trials"],
                                     "percent_r_1", "percent_r_2")
    else:
        ICC_percent_r_hvm = None

    metrics["stats"] = {"ICC_LT_hvm": ICC_looking_time_hvm,
                        "ICC_LT_hvh": ICC_looking_time_hvh,
                        "ICC_PR_hvm": ICC_percent_r_hvm,
                        "ICC_PR_hvh": ICC_percent_r_hvh
                        }

    logging.info("session level stats")

    metrics["human1_vs_machine_session"] = compare_uncollapsed_coding_files(human1_uncol,
                                                                            machine_uncol,
                                                                            [[0, max(end1, mend)]],
                                                                            valid_class_num=valid_class_num)
    metrics["human1_vs_smachine_session"] = compare_uncollapsed_coding_files(human1_uncol,
                                                                             special_machine,
                                                                             [[0, max(end1, mend)]],
                                                                             valid_class_num=valid_class_num)
    if human_coding_file2:
        metrics["human1_vs_human2_session"] = compare_uncollapsed_coding_files(human1_uncol,
                                                                               human2_uncol,
                                                                               [[0, max(end1, end2)]])
    return metrics


def calc_ICC(metrics, dependant_measure1, dependant_measure2):
    """
    calculates ICC3 for single fixesd raters (https://pingouin-stats.org/generated/pingouin.intraclass_corr.html)
    :param metrics: dictionary of results upon trials
    :param dependant_measure1: the measure to calculate ICC over by coder1
    :param dependant_measure2: the measure to calculate ICC over by coder2
    :return: ICC3 metric
    """
    ratings = np.array([[x[dependant_measure1], x[dependant_measure2]] for x in metrics])
    valid_ratings = np.all(~np.isnan(ratings), axis=1)
    ratings = ratings[valid_ratings]
    if len(ratings) >= 5:
        n_trials = len(ratings)
        ratings = ratings.reshape(-1)
        trial_n = np.repeat(np.arange(0, n_trials, 1), 2)
        coders = np.array([[1, 2] for _ in range(n_trials)]).reshape(-1)
        df = pd.DataFrame({'trial_n': trial_n,
                           'coders': coders,
                           'ratings': ratings})
        icc = pg.intraclass_corr(data=df, targets='trial_n', raters='coders', ratings='ratings')
        LT_ICC = icc["ICC"][2]
    else:
        LT_ICC = np.nan
    return LT_ICC


def pick_interesting_frames(coding1, coding2, machine_code):
    """
    given 3 coding files, selects interesting frames where annotators agree and disagree
    :param coding1: human coding1
    :param coding2: human coding2
    :param machine_code: machine coding
    :return: the relevant frame indices
    """
    start1 = np.where(coding1 > 0)[0][0]
    start2 = np.where(coding2 > 0)[0][0]
    start3 = np.where(machine_code > 0)[0][0]
    start = max(max(start1, start2), start3)
    end1 = np.where(coding1 > 0)[0][-1]
    end2 = np.where(coding2 > 0)[0][-1]
    end3 = np.where(machine_code > 0)[0][-1]
    end = min(min(end1, end2), end3)
    default_tuple = (np.array([np.NAN]),)
    agree_away = np.where((machine_code[start:end] == 0) & (coding1[start:end] == 0) & (coding2[start:end] == 0))
    agree_left = np.where((machine_code[start:end] == 1) & (coding1[start:end] == 1) & (coding2[start:end] == 1))
    agree_right = np.where((machine_code[start:end] == 2) & (coding1[start:end] == 2) & (coding2[start:end] == 2))

    disagree_mleft_haway = np.where((machine_code[start:end] == 1) & (coding1[start:end] == 0) & (coding2[start:end] == 0))
    disagree_mright_haway = np.where((machine_code[start:end] == 2) & (coding1[start:end] == 0) & (coding2[start:end] == 0))
    disagree_mx_haway = (np.concatenate((disagree_mleft_haway[0], disagree_mright_haway[0])),)
    disagree_mleft_hright = np.where((machine_code[start:end] == 1) & (coding1[start:end] == 2) & (coding2[start:end] == 2))
    disagree_mright_hleft = np.where((machine_code[start:end] == 2) & (coding1[start:end] == 1) & (coding2[start:end] == 1))

    invalidm_away = np.where((machine_code[start:end] < 0) & (coding1[start:end] == 0) & (coding2[start:end] == 0))
    invalidm_left = np.where((machine_code[start:end] < 0) & (coding1[start:end] == 1) & (coding2[start:end] == 1))
    invalidm_right = np.where((machine_code[start:end] < 0) & (coding1[start:end] == 2) & (coding2[start:end] == 2))

    if agree_away[0].size == 0:
        agree_away = default_tuple
    if agree_left[0].size == 0:
        agree_left = default_tuple
    if agree_right[0].size == 0:
        agree_right = default_tuple

    if disagree_mx_haway[0].size == 0:
        disagree_mx_haway = default_tuple
    if disagree_mleft_hright[0].size == 0:
        disagree_mleft_hright = default_tuple
    if disagree_mright_hleft[0].size == 0:
        disagree_mright_hleft = default_tuple

    if invalidm_away[0].size == 0:
        invalidm_away = default_tuple
    if invalidm_left[0].size == 0:
        invalidm_left = default_tuple
    if invalidm_right[0].size == 0:
        invalidm_right = default_tuple

    selected_frames = [np.random.permutation(agree_left[0])[0],
                       np.random.permutation(agree_away[0])[0],
                       np.random.permutation(agree_right[0])[0],
                       np.random.permutation(invalidm_left[0])[0],
                       np.random.permutation(invalidm_away[0])[0],
                       np.random.permutation(invalidm_right[0])[0],
                       np.random.permutation(disagree_mright_hleft[0])[0],
                       np.random.permutation(disagree_mx_haway[0])[0],
                       np.random.permutation(disagree_mleft_hright[0])[0]]
    selected_frames = np.array(selected_frames)
    selected_frames += start
    return selected_frames


def select_frames_from_video(ID, video_folder, frames):
    """
    selects 9 frames from a video for display
    :param ID: the video id (no extension)
    :param video_folder: the raw video folder
    :param start: where annotation begins
    :param end: where annotation ends
    :return: an image grid of 9 frames and the corresponding frame numbers
    """
    imgs = [None]*9
    filled_counter = 0
    # if frames is None:
    #     frame_selections = np.random.choice(np.arange(0, end // 2), size=9, replace=False)
    # else:
    frame_selections = frames
    invalid_frames = [i for i, x in enumerate(frame_selections) if np.isnan(x)]
    filled_counter += len(invalid_frames)
    for video_file in Path(video_folder).glob("*"):
        if ID in video_file.name:
            cap = cv2.VideoCapture(str(video_file))
            ret, frame = cap.read()
            h, w = frame.shape[0:2]
            frame_counter = 0
            while ret:
                if frame_counter in frame_selections:
                    index = np.where(frame_selections == frame_counter)[0].item()
                    imgs[index] = frame[..., ::-1]
                    filled_counter += 1
                    if filled_counter == 9:
                        for inv_frame in invalid_frames:
                            imgs[inv_frame] = 255*np.ones((h, w, 3), dtype=int)
                        imgs_np = np.array(imgs)
                        imgs_np = make_gridview(imgs_np)
                        return imgs_np, frame_selections
                ret, frame = cap.read()
                frame_counter += 1
    raise IndexError


def sample_luminance(ID, raw_video_folder, start, end, num_samples=100):
    """
    extract average luminance from some frames sampled from a video
    see https://en.wikipedia.org/wiki/Relative_luminance
    :param ID: ID of video
    :param args: video folder
    :param start: frame to start sampling from
    :param end: frame to end sampling from
    :param num_samples: how many samples should we use
    :return: average luminance
    """
    lum_means = []
    for video_file in raw_video_folder.glob("*"):
        if ID in video_file.stem:
            cap = cv2.VideoCapture(str(video_file))
            cur_frame = 0
            frames_ids = np.random.choice(np.arange(start, end//4), size=num_samples, replace=False)
            while len(lum_means) < len(frames_ids):
                ret, frame = cap.read()
                if cur_frame in frames_ids:
                    b, g, r = cv2.split(frame)
                    b = (b / 255) ** 2.2
                    g = (g / 255) ** 2.2
                    r = (r / 255) ** 2.2
                    lum_image = 0.2126 * r + 0.7152 * g + 0.0722 * b
                    lum_means.append(np.mean(lum_image))
                    logging.info("{} / {} samples collected for luminance".format(len(lum_means), num_samples))
                cur_frame += 1
    return np.mean(lum_means)


def session_frame_by_frame_plot(target_ID, metric, session_path):
    skip = 10
    GRAPH_CLASSES = ["left", "right", "away", "invalid"]
    plt.rc('font', size=16)
    fig, ax = plt.subplots(figsize=(16, 6))
    timeline = ax
    # timeline.set_aspect(0.5)
    start1, end1 = metric["human1_vs_human2_session"]["start"], \
                   metric["human1_vs_human2_session"]["end"]
    start2, end2 = metric["human1_vs_machine_session"]["start"], \
                   metric["human1_vs_machine_session"]["end"]
    start = min(start1, start2)
    end = max(end1, end2)
    # plt.suptitle('Video ID: {}, Frames: ({} - {})'.format(target_ID + ".mp4", str(start), str(end)))
    times1 = metric["human1_vs_human2_session"]["times_coding2"]
    times2 = metric["human1_vs_human2_session"]["times_coding1"]
    times3 = metric["human1_vs_machine_session"]["times_coding2"]
    times = [times1, times2, times3]
    video_label = ["Human 2", "Human 1", "iCatcher+"]
    trial_times = [x["end"] for x in metric["human1_vs_human2_trials"]]
    # coding1 = all_metrics[target_ID]["human1_vs_human2_session"]['raw_coding1']
    # coding2 = all_metrics[target_ID]["human1_vs_human2_session"]['raw_coding2']
    # machine_code = all_metrics[target_ID]["human1_vs_machine_session"]['raw_coding2']
    # intersting_frames = pick_interesting_frames(coding1, coding2, machine_code)
    # valid_interesting_frames = [x for x in intersting_frames if not np.isnan(x)]
    vlines_handle = timeline.vlines(trial_times, -1, 3, ls='--', color='k', label="trial end")
    # vlines_selected_frames_handle = timeline.vlines(valid_interesting_frames, -1, 3, ls="solid", color='red', label="selected frames")
    # for i, x in enumerate(valid_interesting_frames):
    #     timeline.text(x, 0, "frame %d" % i, rotation=90, verticalalignment='center')
    # timeline.annotate([str(x) for x in range(len(intersting_frames))],
    #                                        ([0 for _ in range(len(intersting_frames))], intersting_frames))
    timeline.set_xlim([start, end])
    timeline.set_xlabel("Frame #")
    for j, vid_label in enumerate(video_label):
        for label in GRAPH_CLASSES:
            timeline.barh(vid_label, skip, left=times[j][label][::skip],
                          height=1, label=label,
                          color=label_to_color(label))
    artists = [Patch(facecolor=label_to_color("away"), edgecolor='black', lw=1, label="Away"),
               Patch(facecolor=label_to_color("left"), edgecolor='black', lw=1, label="Left"),
               Patch(facecolor=label_to_color("right"), edgecolor='black', lw=1, label="Right"),
               Patch(facecolor=label_to_color("invalid"), edgecolor='black', lw=1, label="Invalid"),
               vlines_handle]  # vlines_selected_frames_handle
    timeline.legend(handles=artists, loc='lower left', bbox_to_anchor=(0.0, 1.01), ncol=5,
                    borderaxespad=0)
    plt.savefig(Path(session_path, 'session_frame_by_frame.pdf'))

    # plt.subplots_adjust(left=0.075, bottom=0.075, right=0.925, top=0.925, wspace=0.2, hspace=0.8)
    plt.cla()
    plt.clf()
    plt.close(fig)


def session_image_collage_plot(target_ID, metric, session_path):
    fig, ax = plt.subplots()
    coding1 = metric["human1_vs_human2_session"]['raw_coding1']
    coding2 = metric["human1_vs_human2_session"]['raw_coding2']
    machine_code = metric["human1_vs_machine_session"]['raw_coding2']
    intersting_frames = pick_interesting_frames(coding1, coding2, machine_code)
    imgs, times = select_frames_from_video(target_ID, args.raw_video_folder,
                                           frames=intersting_frames)
    ax.imshow(imgs)
    # ax.set_xticks([0.33-(1/6), 0.66-(1/6), 1-(1/6)])
    ax.set_xlabel("H1 & H2")
    ax.set_xticks(np.arange(3) * (imgs.shape[1] / 3) + imgs.shape[1] / 6)
    ax.set_xticklabels(["Left", "Away", "Right"])
    ax.set_yticks(np.arange(3) * (imgs.shape[0] / 3) + imgs.shape[0] / 6)
    ax.set_axisbelow(False)
    # ax.set_yticks([0.33-(1/6), 0.66-(1/6), 1-(1/6)])
    # ax.set_yticklabels(["iCatcher+: Correct " + u"\u263A", "iCatcher+: Invalid 😐", "iCatcher+: Wrong " + u"\u2639"])
    ax.set_yticklabels(["Correct", "Invalid", "Incorrect"])
    ax.set_ylabel("iCatcher+")
    # ax.set_ylim(imgs.shape[0] - 0.5, -0.5)
    # ax.set_xlim(imgs.shape[0] - 0.5, -0.5)
    fig.tight_layout()
    # ax.set_axis_off()
    plt.savefig(Path(session_path, 'frame_gallery.pdf'))
    # plt.subplots_adjust(left=0.075, bottom=0.075, right=0.925, top=0.925, wspace=0.2, hspace=0.8)
    plt.cla()
    plt.clf()
    plt.close(fig)


def session_agreement_plot(target_ID, metric, session_path):
    plt.rc('font', size=16)
    fig, ax = plt.subplots(figsize=(6, 8))
    inference = ["human1_vs_human2_session", "human1_vs_machine_session"]
    agreements = [metric[entry]['agreement'] * 100 for entry in inference]
    ax.bar(range(len(inference)), agreements, color="black", width=0.8)
    ax.set_xticks(range(len(inference)))
    ax.set_xticklabels(["H1-H2", "H1-M"])
    ax.set_ylim([0, 100])
    ax.set_ylabel("Percent Agreement")
    plt.savefig(Path(session_path, 'agreement.pdf'), bbox_inches='tight')
    # plt.subplots_adjust(left=0.075, bottom=0.075, right=0.925, top=0.925, wspace=0.2, hspace=0.8)
    plt.cla()
    plt.clf()
    plt.close(fig)


def session_scatter_plot(target_ID, metric, session_path):
    #  looking time plot
    plt.rc('font', size=16)
    fig = plt.figure(figsize=(12, 6))
    lt_scatter = fig.add_subplot(1, 2, 1)
    lt_scatter.set_box_aspect(1)
    lt_scatter.plot([0, 1], [0, 1], transform=lt_scatter.transAxes, color="black", label="Ideal trend")
    fps = 30
    x_target = [x["looking_time_1"] / fps for x in metric["human1_vs_human2_trials"]]
    y_target = [x["looking_time_2"] / fps for x in metric["human1_vs_human2_trials"]]
    y2_target = [x["looking_time_2"] / fps for x in metric["human1_vs_machine_trials"]]
    lt_scatter.scatter(x_target, y_target, color=label_to_color("lblue"),
                       label='H1-H2', marker="o", s=150)
    lt_scatter.scatter(x_target, y2_target, color=label_to_color("lorange"),
                       label='H1-M', marker="^", s=150)
    lt_scatter.set_xlabel("Looking Time (H1)")
    lt_scatter.set_ylabel("Looking Time")
    lt_scatter.set_title("Looking time [s]")
    lt_scatter.legend()  # loc='upper left'
    lt_scatter.set_xlim([0, max(x_target)+1])
    lt_scatter.set_ylim([0, max(y_target + y2_target)+1])

    # %R plot
    pr_scatter = fig.add_subplot(1, 2, 2)
    pr_scatter.set_box_aspect(1)
    pr_scatter.plot([0, 1], [0, 1], transform=pr_scatter.transAxes, color="black", label="Ideal trend")
    x_target = [x["percent_r_1"] * 100 for x in metric["human1_vs_human2_trials"]]
    y_target = [x["percent_r_2"] * 100 for x in metric["human1_vs_human2_trials"]]
    y2_target = [x["percent_r_2"] * 100 for x in metric["human1_vs_machine_trials"]]
    pr_scatter.scatter(x_target, y_target, color=label_to_color("lblue"),
                       label='H1-H2', marker="o", s=150)
    pr_scatter.scatter(x_target, y2_target, color=label_to_color("lorange"),
                       label='H1-M', marker="^", s=150)
    pr_scatter.set_xlabel("Percent Right (H1)")
    pr_scatter.set_ylabel("Percent Right")
    pr_scatter.set_title("Percent Right")
    pr_scatter.legend()  # loc='lower center'
    pr_scatter.set_xlim([0, 100])
    pr_scatter.set_ylim([0, 100])

    plt.savefig(Path(session_path, 'scatter_plots.pdf'))
    plt.cla()
    plt.clf()
    plt.close(fig)


def generate_session_plots(sorted_IDs, all_metrics, args, anonymous=False):
    sessions_path = Path(args.output_folder, "per_session_plots")
    for i, id in enumerate(sorted_IDs):
        print("{} = {}".format(i, id))
    for i, target_ID in enumerate(tqdm(sorted_IDs)):
        if anonymous:
            session_path = Path(sessions_path, "{:02d}".format(i))
        else:
            session_path = Path(sessions_path, "{:02d}_".format(i) + target_ID)
        session_path.mkdir(exist_ok=True, parents=True)
        session_frame_by_frame_plot(target_ID, all_metrics[target_ID], session_path)
        session_agreement_plot(target_ID, all_metrics[target_ID], session_path)
        session_scatter_plot(target_ID, all_metrics[target_ID], session_path)
        if args.raw_dataset_type != "just_annotations":
            session_image_collage_plot(target_ID, all_metrics[target_ID], session_path)


def generate_collage_plot2(sorted_IDs, all_metrics, save_path):
    """
    plots one image with various selected stats
    :param sorted_IDs: ids of videos sorted by accuracy score
    :param all_metrics: all metrics per video
    :param save_path: where to save the image
    :return:
    """
    classes = {"away": 0, "left": 1, "right": 2}
    # fig, axs = plt.subplots(3, 2, figsize=(10, 12))
    fig = plt.figure(figsize=(10, 12))

    # confusion matrix
    conf_mat_h2h = fig.add_subplot(3, 3, 1)  # three rows, three columns
    total_confusion_h2h = np.sum([all_metrics[ID]["human1_vs_human2_session"]["confusion_matrix"] for ID in sorted_IDs],
                                 axis=0)
    total_confusion_h2h /= np.sum(total_confusion_h2h, 0, keepdims=True)  # normalize column-wise
    sns.heatmap(total_confusion_h2h, ax=conf_mat_h2h, vmin=0, vmax=1, annot=True, fmt='.2%', cbar=False, cmap='Blues')
    conf_mat_h2h.set_xticklabels(['away', 'left', 'right'])
    conf_mat_h2h.set_yticklabels(['away', 'left', 'right'])
    conf_mat_h2h.set_xlabel('Human 1')
    conf_mat_h2h.set_ylabel('Human 2')
    conf_mat_h2h.set_title('Human 1 vs Human 2')

    # confusion matrix 2
    conf_mat_h2h = fig.add_subplot(3, 3, 2)
    total_confusion_h2h = np.sum([all_metrics[ID]["human1_vs_machine_session"]["confusion_matrix"] for ID in sorted_IDs],
                                 axis=0)
    total_confusion_h2h /= np.sum(total_confusion_h2h, 0, keepdims=True)  # normalize column-wise
    sns.heatmap(total_confusion_h2h, ax=conf_mat_h2h, vmin=0, vmax=1, annot=True, fmt='.2%', cbar=False, cmap='Blues')
    conf_mat_h2h.set_xticklabels(['away', 'left', 'right'])
    conf_mat_h2h.set_yticklabels(['away', 'left', 'right'])
    conf_mat_h2h.set_xlabel('Human 1')
    conf_mat_h2h.set_ylabel('Model')
    conf_mat_h2h.set_title('Human 1 vs Model')

    # confusion matrix 3
    conf_mat_h2h = fig.add_subplot(3, 3, 3)
    total_confusion_h2h = np.sum([all_metrics[ID]["human1_vs_smachine_session"]["confusion_matrix"] for ID in sorted_IDs],
                                 axis=0)
    total_confusion_h2h /= np.sum(total_confusion_h2h, 0, keepdims=True)  # normalize column-wise
    sns.heatmap(total_confusion_h2h, ax=conf_mat_h2h, vmin=0, vmax=1, annot=True, fmt='.2%', cbar=False, cmap='Blues')
    conf_mat_h2h.set_xticklabels(['away', 'left', 'right'])
    conf_mat_h2h.set_yticklabels(['away', 'left', 'right'])
    conf_mat_h2h.set_xlabel('Human 1')
    conf_mat_h2h.set_ylabel('Model')
    conf_mat_h2h.set_title("Human 1 vs Model, invalid frames = away")

    # LR ICC plot
    lr_icc_scatter = fig.add_subplot(3, 3, 4)
    lr_icc_scatter.plot([0, 1], [0, 1], transform=lr_icc_scatter.transAxes, color="black", label="Ideal trend")
    lr_icc_scatter.set_xlim([0, 1])
    lr_icc_scatter.set_ylim([0, 1])
    x_target = [all_metrics[ID]["stats"]["ICC_LT_hvh"] for ID in sorted_IDs]
    y_target = [all_metrics[ID]["stats"]["ICC_LT_hvm"] for ID in sorted_IDs]
    lr_icc_scatter.scatter(x_target, y_target, color=label_to_color("lblue"),
                       label='Session')
    lr_icc_scatter.set_xlabel("Human 1 vs Human 2")
    lr_icc_scatter.set_ylabel("Human 1 vs Model")
    lr_icc_scatter.set_title("Looking Time ICC")
    lr_icc_scatter.legend(loc='upper left')

    # PR ICC plot
    pr_icc_scatter = fig.add_subplot(3, 3, 5)
    pr_icc_scatter.plot([0, 1], [0, 1], transform=pr_icc_scatter.transAxes, color="black", label="Ideal trend")
    pr_icc_scatter.set_xlim([0, 1])
    pr_icc_scatter.set_ylim([0, 1])
    x_target = [all_metrics[ID]["stats"]["ICC_PR_hvh"] for ID in sorted_IDs]
    y_target = [all_metrics[ID]["stats"]["ICC_PR_hvm"] for ID in sorted_IDs]
    pr_icc_scatter.scatter(x_target, y_target, color=label_to_color("lblue"),
                        label='Session')
    pr_icc_scatter.set_xlabel("Human 1 vs Human 2")
    pr_icc_scatter.set_ylabel("Human 1 vs Model")
    pr_icc_scatter.set_title("Percent Right ICC")
    pr_icc_scatter.legend(loc='upper left')

    # LT plot
    lt_scatter = fig.add_subplot(3, 3, 6)
    lt_scatter.plot([0, 1], [0, 1], transform=lt_scatter.transAxes, color="black", label="Ideal trend")

    x_target = []
    y_target = []
    for ID in sorted_IDs:
        x_target += [x["looking_time_1"]/30 for x in all_metrics[ID]["human1_vs_machine_trials"]]
        y_target += [x["looking_time_2"]/30 for x in all_metrics[ID]["human1_vs_machine_trials"]]
    maxi = np.max(x_target + y_target)
    lt_scatter.set_xlim([0, maxi])
    lt_scatter.set_ylim([0, maxi])

    lt_scatter.scatter(x_target, y_target, color=label_to_color("lorange"),
                       label='Trial', alpha=0.3)
    lt_scatter.set_xlabel("Human 1")
    lt_scatter.set_ylabel("Model")
    lt_scatter.set_title("Looking time [s]")
    lt_scatter.legend(loc='upper left')

    # %R plot
    pr_scatter = fig.add_subplot(3, 3, 7)
    pr_scatter.plot([0, 1], [0, 1], transform=pr_scatter.transAxes, color="black", label="Ideal trend")
    pr_scatter.set_xlim([0, 100])
    pr_scatter.set_ylim([0, 100])
    x_target = []
    y_target = []
    for ID in sorted_IDs:
        x_target += [x["percent_r_1"] * 100 for x in all_metrics[ID]["human1_vs_machine_trials"]]
        y_target += [x["percent_r_2"] * 100 for x in all_metrics[ID]["human1_vs_machine_trials"]]
    pr_scatter.scatter(x_target, y_target, color=label_to_color("lorange"),
                       label='Trial', alpha=0.3)
    pr_scatter.set_xlabel("Human 1")
    pr_scatter.set_ylabel("Model")
    pr_scatter.set_title("Percent Right")
    pr_scatter.legend(loc='lower center')

    # percent agreement plot
    pa_scatter = fig.add_subplot(3, 3, 8)
    pa_scatter.plot([0, 1], [0, 1], transform=pa_scatter.transAxes, color="black", label="Ideal trend")
    pa_scatter.set_xlim([0, 100])
    pa_scatter.set_ylim([0, 100])
    x_target = []
    y_target = []
    for ID in sorted_IDs:
        x_target += [x["agreement"] * 100 for x in all_metrics[ID]["human1_vs_human2_trials"]]
        y_target += [x["agreement"] * 100 for x in all_metrics[ID]["human1_vs_machine_trials"]]
    pa_scatter.scatter(x_target, y_target, color=label_to_color("lorange"),
                       label='Trial', alpha=0.3)
    pa_scatter.set_xlabel("Human 1 vs Human 2")
    pa_scatter.set_ylabel("Human 1 vs Model")
    pa_scatter.set_title("Percent Agreement")
    pa_scatter.legend(loc='upper left')

    plt.subplots_adjust(left=0.1, bottom=0.075, right=0.9, top=0.925, wspace=0.5, hspace=0.5)
    plt.savefig(Path(save_path, "collage2.png"))
    plt.cla()
    plt.clf()
    plt.close(fig)


def generate_barplot(sorted_IDs, all_metrics, save_path):
    plt.rc('font', size=13)
    fig, ax = plt.subplots()
    agreement_hvh = [all_metrics[ID]["human1_vs_human2_session"]['agreement'] for ID in sorted_IDs]
    mean_agreement_hvh, std_agreement_hvh1, std_agreement_hvh2 = bootstrap(agreement_hvh)
    # mean_agreement_hvh = np.mean(agreement_hvh)
    # std_agreement_hvh = np.std(agreement_hvh)
    agreement_hvm = [all_metrics[ID]["human1_vs_machine_session"]['agreement'] for ID in sorted_IDs]
    mean_agreement_hvm, std_agreement_hvm1, std_agreement_hvm2 = bootstrap(agreement_hvm)
    # mean_agreement_hvm = np.mean(agreement_hvm)
    # std_agreement_hvm = np.std(agreement_hvm)
    kappa_hvh = [all_metrics[ID]["human1_vs_human2_session"]['kappa'] for ID in sorted_IDs]
    mean_kappa_hvh, std_kappa_hvh1, std_kappa_hvh2 = bootstrap(kappa_hvh)
    # mean_kappa_hvh = np.mean(kappa_hvh)
    # std_kappa_hvh = np.std(kappa_hvh)
    kappa_hvm = [all_metrics[ID]["human1_vs_machine_session"]['kappa'] for ID in sorted_IDs]
    mean_kappa_hvm, std_kappa_hvm1, std_kappa_hvm2 = bootstrap(kappa_hvm)
    # mean_kappa_hvm = np.mean(kappa_hvm)
    # std_kappa_hvm = np.std(kappa_hvm)
    icc_lt_hvh = [all_metrics[ID]["stats"]["ICC_LT_hvh"] for ID in sorted_IDs]
    mean_icc_lt_hvh, std_icc_lt_hvh1, std_icc_lt_hvh2 = bootstrap(icc_lt_hvh)
    # mean_icc_lt_hvh = np.mean(icc_lt_hvh)
    # std_icc_lt_hvh = np.std(icc_lt_hvh)
    icc_lt_hvm = [all_metrics[ID]["stats"]["ICC_LT_hvm"] for ID in sorted_IDs]
    mean_icc_lt_hvm, std_icc_lt_hvm1, std_icc_lt_hvm2 = bootstrap(icc_lt_hvm)
    # mean_icc_lt_hvm = np.mean(icc_lt_hvm)
    # std_icc_lt_hvm = np.std(icc_lt_hvm)
    icc_pr_hvh = [all_metrics[ID]["stats"]["ICC_PR_hvh"] for ID in sorted_IDs]
    mean_icc_pr_hvh, std_icc_pr_hvh1, std_icc_pr_hvh2 = bootstrap(icc_pr_hvh)
    # mean_icc_pr_hvh = np.mean(icc_pr_hvh)
    # std_icc_pr_hvh = np.std(icc_pr_hvh)
    icc_pr_hvm = [all_metrics[ID]["stats"]["ICC_PR_hvm"] for ID in sorted_IDs]
    mean_icc_pr_hvm, std_icc_pr_hvm1, std_icc_pr_hvm2 = bootstrap(icc_pr_hvm)
    # mean_icc_pr_hvm = np.mean(icc_pr_hvm)
    # std_icc_pr_hvm = np.std(icc_pr_hvm)
    x = np.arange(4)
    width = 0.35  # the width of the bars
    ydata1 = np.array([mean_agreement_hvh, mean_kappa_hvh, mean_icc_lt_hvh, mean_icc_pr_hvh])
    yerr1 = np.array([(std_agreement_hvh1, std_agreement_hvh2), (std_kappa_hvh1, std_kappa_hvh2),
                      (std_icc_lt_hvh1, std_icc_lt_hvh2), (std_icc_pr_hvh1, std_icc_pr_hvh2)])
    yerr1 = np.abs(yerr1 - ydata1[:, None])
    ydata2 = np.array([mean_agreement_hvm, mean_kappa_hvm, mean_icc_lt_hvm, mean_icc_pr_hvm])
    yerr2 = np.array([(std_agreement_hvm1, std_agreement_hvm2), (std_kappa_hvm1, std_kappa_hvm2),
                      (std_icc_lt_hvm1, std_icc_lt_hvm2), (std_icc_pr_hvm1, std_icc_pr_hvm2)])
    yerr2 = np.abs(yerr2 - ydata2[:, None])
    rects1 = ax.bar(x - (width / 2), ydata1,
                    yerr=yerr1.T, width=width,
                    label='Human-Human', align='center', ecolor='black', capsize=10)
    rects2 = ax.bar(x + (width / 2), ydata2,
                    yerr=yerr2.T, width=width,
                    label='Human-Model', align='center', ecolor='black', capsize=10)
    labels = ['% Agree', 'Cohen\'s Kappa', 'ICC (LT)', 'ICC (PR)']
    ax.set_xticks(x)
    ax.set_yticks(np.arange(0, 1.2, step=0.2))
    ax.set_xticklabels(labels)  # , rotation=-45
    ax.bar_label(rects1, fmt='%.2f', padding=3, label_type="center")
    ax.bar_label(rects2, fmt='%.2f', padding=3, label_type="center")
    ax.legend(loc='lower right')
    fig.tight_layout()
    plt.savefig(str(Path(save_path, 'dataset_bar_plots_with_error.pdf')))
    plt.cla()
    plt.clf()
    plt.close(fig)


def generate_confusion_matrices(sorted_IDs, all_metrics, args):
    save_path = args.output_folder
    # widths = [1, 1, 1, 1]
    # heights = [1]
    # gs_kw = dict(width_ratios=widths, height_ratios=heights)
    # fig, axs = plt.subplots()  # 1, 4, gridspec_kw=gs_kw, figsize=(24.0, 8.0),
    plt.rc('font', size=16)
    fig = plt.figure(figsize=(10, 10))
    # conf_mat_h2h, conf_mat2_h2h, conf_mat_h2m, conf_mat2_h2m = axs  # won't work with single video...
    conf_mat_h2h = fig.add_subplot(2, 2, 1)
    total_confusion_h2h = np.sum([all_metrics[ID]["human1_vs_human2_session"]["confusion_matrix"] for ID in sorted_IDs],
                                 axis=0)
    total_confusion_h2h /= np.sum(total_confusion_h2h, 0, keepdims=True)  # normalize column-wise
    sns.heatmap(total_confusion_h2h, ax=conf_mat_h2h, vmin=0, vmax=1, annot=True, fmt='.2%', cbar=False, cmap='Blues',
                annot_kws={"size": 24})
    conf_mat_h2h.set_xticklabels(['away', 'left', 'right'])
    conf_mat_h2h.set_yticklabels(['away', 'left', 'right'])
    conf_mat_h2h.set_xlabel('Human 1')
    conf_mat_h2h.set_ylabel('Human 2')
    conf_mat_h2h.set_box_aspect(1)

    conf_mat2_h2h = fig.add_subplot(2, 2, 2)
    total_confusion2_h2h = np.sum([all_metrics[ID]["human1_vs_human2_session"]["confusion_matrix2"] for ID in sorted_IDs],
                                  axis=0)
    total_confusion2_h2h /= np.sum(total_confusion2_h2h, 0, keepdims=True)  # normalize column-wise
    sns.heatmap(total_confusion2_h2h, ax=conf_mat2_h2h, vmin=0, vmax=1, annot=True, fmt='.2%', cbar=False, cmap='Blues',
                annot_kws={"size": 26})
    if args.raw_dataset_type == "cali-bw":
        conf_mat2_h2h.set_xticklabels(['off*', 'on'])
        conf_mat2_h2h.set_yticklabels(['off*', 'on'])
    else:
        conf_mat2_h2h.set_xticklabels(['off', 'on'])
        conf_mat2_h2h.set_yticklabels(['off', 'on'])
    conf_mat2_h2h.set_xlabel('Human 1')
    conf_mat2_h2h.set_ylabel('Human 2')
    conf_mat2_h2h.set_box_aspect(1)

    conf_mat_h2m = fig.add_subplot(2, 2, 3)
    total_confusion_h2m = np.sum([all_metrics[ID]["human1_vs_machine_session"]["confusion_matrix"] for ID in sorted_IDs],
                                 axis=0)
    total_confusion_h2m /= np.sum(total_confusion_h2m, 0, keepdims=True)  # normalize column-wise
    sns.heatmap(total_confusion_h2m, ax=conf_mat_h2m, vmin=0, vmax=1, annot=True, fmt='.2%', cbar=False, cmap='Blues',
                annot_kws={"size": 24})
    conf_mat_h2m.set_xticklabels(['away', 'left', 'right'])
    conf_mat_h2m.set_yticklabels(['away', 'left', 'right'])
    conf_mat_h2m.set_xlabel('Human 1')
    conf_mat_h2m.set_ylabel('iCatcher+')
    conf_mat_h2m.set_box_aspect(1)

    conf_mat2_h2m = fig.add_subplot(2, 2, 4)
    total_confusion2_h2m = np.sum([all_metrics[ID]["human1_vs_machine_session"]["confusion_matrix2"] for ID in sorted_IDs],
                                 axis=0)
    total_confusion2_h2m /= np.sum(total_confusion2_h2m, 0, keepdims=True)  # normalize column-wise
    sns.heatmap(total_confusion2_h2m, ax=conf_mat2_h2m, vmin=0, vmax=1, annot=True, fmt='.2%', cbar=False, cmap='Blues',
                annot_kws={"size": 26})
    if args.raw_dataset_type == "cali-bw":
        conf_mat2_h2m.set_xticklabels(['off*', 'on'])
        conf_mat2_h2m.set_yticklabels(['off*', 'on'])
    else:
        conf_mat2_h2m.set_xticklabels(['off', 'on'])
        conf_mat2_h2m.set_yticklabels(['off', 'on'])
    conf_mat2_h2m.set_xlabel('Human 1')
    conf_mat2_h2m.set_ylabel('iCatcher+')
    conf_mat2_h2m.set_box_aspect(1)

    fig.tight_layout()
    # ax.set_axis_off()
    plt.savefig(str(Path(save_path, 'dataset_confusion_matrices.pdf')))
    plt.cla()
    plt.clf()
    plt.close(fig)


def generate_agreement_scatter(sorted_IDs, all_metrics, args, multi_dataset=False):
    save_path = args.output_folder
    plt.rc('font', size=16)
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], transform=ax.transAxes, color="black")
    x_target = [all_metrics[ID]["human1_vs_human2_session"]["agreement"] for ID in sorted_IDs]
    y_target = [all_metrics[ID]["human1_vs_machine_session"]["agreement"] for ID in sorted_IDs]
    if args.raw_dataset_type == "cali-bw":
        primary_label = "California-BW Videos"
        secondary_label = "Lookit Videos"
        np.savez("cali-bw_agreement", x_target, y_target)
    elif args.raw_dataset_type == "senegal":
        primary_label = "Senegal Videos"
        secondary_label = "Lookit Videos"
        np.savez("senegal_agreement", x_target, y_target)
    elif args.raw_dataset_type == "lookit":
        primary_label = "Lookit Videos"
        secondary_label = "California-BW Videos"
        third_label = "Senegal Videos"
    else:
        primary_label = "My Dataset"
        secondary_label = "placeholder"
        third_label = "placeholder"
    ax.scatter(x_target, y_target,
               color=label_to_color("vlblue"), label=primary_label, alpha=0.5, s=40, marker="o")
    meanx, confx1, confx2 = bootstrap(x_target)
    meany, confy1, confy2 = bootstrap(y_target)
    ax.errorbar(meanx, meany,
                xerr=np.array([meanx - confx1, confx2 - meanx])[:, None],
                yerr=np.array([meany - confy1, confy2 - meany])[:, None],
                barsabove=True,
                color="k", markerfacecolor=label_to_color("vblue"),
                linewidth=1, marker='o', capsize=3, ms=10)  # ms=40
    minx = min(x_target)
    miny = min(y_target)
    plot_name = 'dataset_agreement_scatter.pdf'
    if multi_dataset:
        data = np.load("cali-bw_agreement.npz")
        x_target_2, y_target_2 = data["arr_0"], data["arr_1"]
        minx = min(minx, np.min(x_target_2))
        miny = min(miny, np.min(y_target_2))
        ax.scatter(x_target_2, y_target_2,
                   color=label_to_color("vlgreen"), label=secondary_label, alpha=0.5, s=40, marker="^")
        meanx, confx1, confx2 = bootstrap(x_target_2)
        meany, confy1, confy2 = bootstrap(y_target_2)
        ax.errorbar(meanx, meany,
                    xerr=np.array([meanx - confx1, confx2 - meanx])[:, None],
                    yerr=np.array([meanx - confx1, confx2 - meanx])[:, None],
                    barsabove=True,
                    color="k", markerfacecolor=label_to_color("vgreen"),
                    linewidth=1, marker='^', capsize=3, ms=10)  # ms=40
        data = np.load("senegal_agreement.npz")
        x_target_3, y_target_3 = data["arr_0"], data["arr_1"]
        minx = min(minx, np.min(x_target_3))
        miny = min(miny, np.min(y_target_3))
        ax.scatter(x_target_3, y_target_3,
                   color=label_to_color("vlpurple"), label=third_label, alpha=0.5, s=40, marker="s")
        meanx, confx1, confx2 = bootstrap(x_target_3)
        meany, confy1, confy2 = bootstrap(y_target_3)
        ax.errorbar(meanx, meany,
                    xerr=np.array([meanx - confx1, confx2 - meanx])[:, None],
                    yerr=np.array([meanx - confx1, confx2 - meanx])[:, None],
                    barsabove=True,
                    color="k", markerfacecolor=label_to_color("vpurple"),
                    linewidth=1, marker='s', capsize=3, ms=10)  # ms=40
        plot_name = "multi_dataset_agreement_scatter.pdf"
    final_min = min(minx, miny)
    ax.set_xlim([final_min, 1])
    ax.set_ylim([final_min, 1])
    ax.set_xlabel("H-H Percent Agreement")
    ax.set_ylabel("H-M Percent Agreement")
    # ax.set_title("Percent Agreement")
    ax.legend(loc='upper left')

    plt.savefig(str(Path(save_path, plot_name)), bbox_inches='tight')
    plt.cla()
    plt.clf()
    plt.close(fig)


def generate_age_vs_agreement(sorted_IDs, all_metrics, args, video_dataset):
    x = []
    y = []
    for id in sorted_IDs:
        agreement = all_metrics[id]["human1_vs_machine_session"]["agreement"] * 100
        age = video_dataset[id]["child_age"]
        x.append(int(age))
        y.append(agreement)
    plt.rc('font', size=16)
    fig, ax = plt.subplots()
    if args.raw_dataset_type == "cali-bw":
        color = label_to_color("vlgreen")
    elif args.raw_dataset_type == "senegal":
        color = label_to_color("vlpurple")
    else:
        color = label_to_color("vlblue")
    ax = sns.regplot(x=x, y=y, color=color)

    # ax.scatter(x, y,
    #            color=label_to_color("vlblue"), alpha=0.5, s=40, marker="o")
    ax.set_xlabel("Age [months]")
    ax.set_ylabel("Percent Agreement")
    save_path = args.output_folder
    plt.savefig(str(Path(save_path, "agreement_vs_age.pdf")), bbox_inches='tight')
    plt.cla()
    plt.clf()
    plt.close(fig)


def perform_custom_permutation(category_str, unique_labels, inverse):
    if category_str == "child_skin_tone":
        perm = np.array([np.where(unique_labels == "light")[0],
                         np.where(unique_labels == "medium")[0],
                         np.where(unique_labels == "dark")[0]])
    else:
        perm = np.arange(len(unique_labels))
    perm = perm.squeeze()
    new_labels = unique_labels[perm]
    new_inverse = []
    for i in range(len(inverse)):
        new_inverse.append(np.where(perm == inverse[i])[0])
    return new_labels, np.array(new_inverse).squeeze()


def generate_categorial_vs_agreement(sorted_IDs, all_metrics, args, video_dataset, category_str, indi_points=True):
    category = []
    y = []
    for id in sorted_IDs:
        agreement = all_metrics[id]["human1_vs_machine_session"]["agreement"] * 100
        category.append(video_dataset[id][category_str])
        y.append(agreement)
    y = np.array(y)
    labels, inverse = np.unique(category, return_inverse=True)
    labels, inverse = perform_custom_permutation(category_str, labels, inverse)
    data = []
    err = []
    for i in range(len(labels)):
        mean, confb, confu = bootstrap(y[inverse == i])
        data.append(mean)
        err.append((mean - confb, confu - mean))
    plt.rc('font', size=14)
    fig, ax = plt.subplots(figsize=(6, 8))
    if args.raw_dataset_type == "cali-bw":
        color_str = "vlgreen"
    elif args.raw_dataset_type == "senegal":
        color_str = "vlpurple"
    else:
        color_str = "vlblue"
    color = label_to_color(color_str)
    w = 0.8  # bar width
    x = [n for n in range(len(labels))]
    ax.bar(x, data, yerr=np.array(err).T, color=color, capsize=10, width=w)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylim([0, 100])
    ax.set_ylabel("Percent Agreement")
    if indi_points:
        for i in range(len(labels)):
            # distribute scatter randomly across whole width of bar
            ax.scatter(x[i] + np.random.random(y[inverse == i].size) * w - w / 2, y[inverse == i],
                       color="black", zorder=2)
    save_path = args.output_folder
    plt.savefig(str(Path(save_path, "agreement_vs_{}.pdf".format(category_str))), bbox_inches='tight')
    plt.cla()
    plt.clf()
    plt.close(fig)


def generate_posture_vs_agreement(sorted_IDs, all_metrics, args):
    agreements = []
    postures = []
    labels = ["Over shoulder", "Sitting in lap", "Sitting alone", "Other posture"]  # "No Posture"
    for id in sorted_IDs:
        raw_postures = all_metrics[id]["postures"]
        trial_start_times = [x["end"] for x in all_metrics[id]["human1_vs_machine_trials"]]
        trial_start_times.insert(0, all_metrics[id]["human1_vs_machine_trials"][0]["start"])
        trial_hvm_agreement = [x["agreement"] * 100 for x in all_metrics[id]["human1_vs_machine_trials"]]
        for i in range(len(trial_start_times) - 1):  # last time in list is end of last trial
            postures_in_trial = raw_postures[trial_start_times[i]:trial_start_times[i+1]]
            valid_postures = postures_in_trial[postures_in_trial >= 0]
            uni_values = np.unique(valid_postures)
            if len(uni_values) == 1:
                postures.append(uni_values)
                agreements.append(trial_hvm_agreement[i])
    agreements = np.array(agreements)
    postures = np.array(postures).squeeze()
    nan_mask = ~np.isnan(agreements)
    agreements = agreements[nan_mask]
    postures = postures[nan_mask]
    ydata = []
    yerr = []
    for i in range(len(labels)):
        mean, confb, confu = bootstrap(agreements[postures == i])
        ydata.append(mean)
        yerr.append((mean - confb, confu - mean))
    ydata = np.array(ydata)
    yerr = np.array(yerr)
    plt.rc('font', size=14)
    fig, ax = plt.subplots(figsize=(6, 8))
    x = np.arange(len(labels))
    w = 0.8  # bar width
    ax.bar(x, ydata, yerr=yerr.T, color=label_to_color("vlblue"), capsize=10, width=w)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=-45)
    ax.set_ylim([0, 100])
    ax.set_ylabel("Percent Agreement")
    for i in range(len(labels)):
        # distribute scatter randomly across whole width of bar
        ax.scatter(x[i] + np.random.random(agreements[postures == i].size) * w - w / 2, agreements[postures == i],
                   color="black", zorder=2, alpha=0.2)
    save_path = args.output_folder
    plt.savefig(str(Path(save_path, "agreement_vs_posture.pdf")), bbox_inches='tight')
    plt.cla()
    plt.clf()
    plt.close(fig)


def generate_in_out_trial_vs_agreement(sorted_IDs, all_metrics, args):
    agreement_in = []
    agreement_between = []
    for id in sorted_IDs:
        raw_human = all_metrics[id]["human1_vs_machine_session"]["raw_coding1"]
        raw_machine = all_metrics[id]["human1_vs_machine_session"]["raw_coding2"]
        trial_end_times = [x["end"] for x in all_metrics[id]["human1_vs_machine_trials"]]
        between_trial_mask = np.zeros_like(raw_human).astype(bool)
        in_trial_mask = np.zeros_like(raw_human).astype(bool)
        between_indices = np.array([np.arange(x-4, x+5) for x in trial_end_times[:-1]]).reshape(-1)
        in_indices = np.setdiff1d(np.arange(len(raw_human)), between_indices)
        in_indices = np.random.choice(in_indices, size=len(between_indices), replace=False)  # sample to make equal amount
        between_trial_mask[between_indices] = True
        in_trial_mask[in_indices] = True
        mutually_valid_mask = np.logical_and(raw_human >= 0, raw_machine >= 0)
        human_between_trials = raw_human[between_trial_mask & mutually_valid_mask]
        human_in_trials = raw_human[in_trial_mask & mutually_valid_mask]
        machine_between_trials = raw_machine[between_trial_mask & mutually_valid_mask]
        machine_in_trials = raw_machine[in_trial_mask & mutually_valid_mask]
        agreement_between_trials = np.count_nonzero(machine_between_trials == human_between_trials) / len(machine_between_trials)
        agreement_between.append(agreement_between_trials)
        agreement_in_trials = np.count_nonzero(machine_in_trials == human_in_trials) / len(machine_in_trials)
        agreement_in.append(agreement_in_trials)

    agreement_between = np.array(agreement_between)
    agreement_in = np.array(agreement_in)
    t, p, dof = t_test_paired(agreement_in, agreement_between)
    print("t-test (paired) trial_between vs trial_in: t={:.2f}, p={:.8f}, dof={}]".format(t, p, dof))
    in_mean, in_b, in_u = bootstrap(agreement_in)
    between_mean, between_b, between_u = bootstrap(agreement_between)
    plt.rc('font', size=16)
    fig, ax = plt.subplots()
    x = np.arange(2)
    y = np.array([agreement_in, agreement_between])
    primary_label = "Lookit"
    w = 0.35  # the width of the bars
    ydata = np.array([in_mean, between_mean])
    yerr = np.array([(in_b, in_u),
                     (between_b, between_u)])
    yerr = np.abs(yerr - ydata[:, None])
    rects1 = ax.bar(x, ydata,
                    yerr=yerr.T, width=w,
                    label=primary_label, align='center', ecolor='black',
                    color=label_to_color("vlblue"), capsize=10)
    labels = ['H1-M During Trials', 'H1-M Between Trials']
    ax.set_xticks(x)
    ax.set_yticks(np.arange(0, 1.2, step=0.2))
    ax.set_xticklabels(labels)  # , rotation=-45
    ax.bar_label(rects1, fmt='%.2f', padding=3, label_type="center")
    ax.legend(loc='lower right')
    ax.set_ylabel("Agreement")
    for i in range(len(labels)):
        # distribute scatter randomly across whole width of bar
        ax.scatter(x[i] + np.random.random(y[i].size) * w - w / 2, y[i],
                   color="black", zorder=2)
    save_path = args.output_folder
    name = "agreement_vs_in-trial_between-trial.pdf"
    plt.savefig(str(Path(save_path, name)), bbox_inches='tight')
    plt.cla()
    plt.clf()
    plt.close(fig)


def generate_confidence_vs_agreement(sorted_IDs, all_metrics, args, multi_dataset=False):
    # agreement = [all_metrics[id]["human1_vs_machine_session"]["agreement"] * 100 for id in sorted_IDs]
    confidence_correct = []
    confidence_incorrect = []
    for ID in sorted_IDs:
        confidence_correct += [x["confidence_metrics"][0] for x in all_metrics[ID]["human1_vs_machine_trials"]]
        confidence_incorrect += [x["confidence_metrics"][1] for x in all_metrics[ID]["human1_vs_machine_trials"]]

    confidence_correct = np.array(confidence_correct)
    confidence_incorrect = np.array(confidence_incorrect)
    valid_trials_confidence = ~np.isnan(confidence_correct) & ~np.isnan(confidence_incorrect)
    confidence_correct = confidence_correct[valid_trials_confidence]
    confidence_incorrect = confidence_incorrect[valid_trials_confidence]
    if args.raw_dataset_type == "cali-bw":
        primary_label = "California-BW"
        secondary_label = "Lookit"
        np.savez("cali-bw_confidence", confidence_correct, confidence_incorrect)
    elif args.raw_dataset_type == "senegal":
        primary_label = "Senegal"
        secondary_label = "Lookit"
        np.savez("senegal_confidence", confidence_correct, confidence_incorrect)
    elif args.raw_dataset_type == "lookit":
        primary_label = "Lookit"
        secondary_label = "California-BW"
        third_label = "Senegal"
    else:
        primary_label = "Dataset"
        secondary_label = "placeholder"
        third_label = "placeholder"
    plt.rc('font', size=16)
    fig, ax = plt.subplots()
    x = np.arange(2)
    width = 0.2  # the width of the bars
    correct_mean, correct_confb, correct_confu = bootstrap(confidence_correct)
    incorrect_mean, incorrect_confb, incorrect_confu = bootstrap(confidence_incorrect)
    ydata = np.array([correct_mean, incorrect_mean])
    yerr = np.array([(correct_confb, correct_confu),
                     (incorrect_confb, incorrect_confu)])
    yerr = np.abs(yerr - ydata[:, None])
    rects1 = ax.bar(x - width, ydata,
                    yerr=yerr.T, width=width,
                    label=primary_label, align='center', ecolor='black',
                    color=label_to_color("vlblue"), capsize=10)
    if multi_dataset:
        data = np.load("cali-bw_confidence.npz")
        x1, x2 = data["arr_0"], data["arr_1"]
        correct_mean2, correct_confb2, correct_confu2 = bootstrap(x1)
        incorrect_mean2, incorrect_confb2, incorrect_confu2 = bootstrap(x2)
        ydata = np.array([correct_mean2, incorrect_mean2])
        yerr = np.array([(correct_confb2, correct_confu2),
                         (incorrect_confb2, incorrect_confu2)])
        yerr = np.abs(yerr - ydata[:, None])
        rects2 = ax.bar(x, ydata,
                        yerr=yerr.T, width=width,
                        label=secondary_label, align='center', ecolor='black',
                        color=label_to_color("vlgreen"), capsize=10)
        data = np.load("senegal_confidence.npz")
        x1, x2 = data["arr_0"], data["arr_1"]
        correct_mean2, correct_confb2, correct_confu2 = bootstrap(x1)
        incorrect_mean2, incorrect_confb2, incorrect_confu2 = bootstrap(x2)
        ydata = np.array([correct_mean2, incorrect_mean2])
        yerr = np.array([(correct_confb2, correct_confu2),
                         (incorrect_confb2, incorrect_confu2)])
        yerr = np.abs(yerr - ydata[:, None])
        rects3 = ax.bar(x + width, ydata,
                        yerr=yerr.T, width=width,
                        label=third_label, align='center', ecolor='black',
                        color=label_to_color("vlpurple"), capsize=10)
    labels = ['H1-M Agree', 'H1-M Disagree']
    ax.set_xticks(x)
    ax.set_yticks(np.arange(0, 1.2, step=0.2))
    ax.set_xticklabels(labels)  # , rotation=-45
    ax.bar_label(rects1, fmt='%.2f', padding=3, label_type="center")
    if multi_dataset:
        ax.bar_label(rects2, fmt='%.2f', padding=3, label_type="center")
        ax.bar_label(rects3, fmt='%.2f', padding=3, label_type="center")
    ax.legend(loc='lower right')
    ax.set_ylabel("Confidence")
    save_path = args.output_folder
    if multi_dataset:
        name = "multi_dataset_agreement_vs_confidence.pdf"
    else:
        name = "agreement_vs_confidence.pdf"
    plt.savefig(str(Path(save_path, name)), bbox_inches='tight')
    plt.cla()
    plt.clf()
    plt.close(fig)


def generate_transitions_plot(sorted_IDs, all_metrics, args, multi_dataset=False):
    transitions_h1 = [100 * all_metrics[ID]["human1_vs_human2_session"]['n_transitions_1'] /
                      all_metrics[ID]["human1_vs_human2_session"]['valid_frames_1'] for ID in sorted_IDs]
    transitions_h2 = [100 * all_metrics[ID]["human1_vs_human2_session"]['n_transitions_2'] /
                      all_metrics[ID]["human1_vs_human2_session"]['valid_frames_2'] for ID in sorted_IDs]
    transitions_h3 = [100 * all_metrics[ID]["human1_vs_machine_session"]['n_transitions_2'] /
                      all_metrics[ID]["human1_vs_machine_session"]['valid_frames_2'] for ID in sorted_IDs]
    transitions_h1 = np.array(transitions_h1)
    transitions_h2 = np.array(transitions_h2)
    transitions_h3 = np.array(transitions_h3)
    if args.raw_dataset_type == "cali-bw":
        np.savez("cali-bw_transitions_per_100", transitions_h1, transitions_h2, transitions_h3)
        return
    elif args.raw_dataset_type == "senegal":
        np.savez("senegal_transitions_per_100", transitions_h1, transitions_h2, transitions_h3)
        return
    elif args.raw_dataset_type == "lookit":
        primary_label = "Lookit"
        secondary_label = "California-BW"
    else:
        primary_label = "Dataset"
        secondary_label = "placeholder"
    plt.rc('font', size=16)
    fig, ax = plt.subplots()
    x = np.arange(3)
    width = 0.35  # the width of the bars
    transitions_h1_mean, transitions_h1_confb, transitions_h1_confu = bootstrap(transitions_h1)
    transitions_h2_mean, transitions_h2_confb, transitions_h2_confu = bootstrap(transitions_h2)
    transitions_h3_mean, transitions_h3_confb, transitions_h3_confu = bootstrap(transitions_h3)
    ydata = np.array([transitions_h1_mean, transitions_h2_mean, transitions_h3_mean])
    yerr = np.array([(transitions_h1_confb, transitions_h1_confu),
                     (transitions_h2_confb, transitions_h2_confu),
                     (transitions_h3_confb, transitions_h3_confu)])
    yerr = np.abs(yerr - ydata[:, None])
    rects1 = ax.bar(x - width / 2, ydata,
                    yerr=yerr.T, width=width,
                    label=primary_label, color=label_to_color("vlblue"),
                    align='center', ecolor='black', capsize=10)
    if multi_dataset:
        data = np.load("cali-bw_transitions_per_100.npz")
        x1, x2, x3 = data["arr_0"], data["arr_1"], data["arr_2"]
        transitions_h1_mean, transitions_h1_confb, transitions_h1_confu = bootstrap(x1)
        transitions_h2_mean, transitions_h2_confb, transitions_h2_confu = bootstrap(x2)
        transitions_h3_mean, transitions_h3_confb, transitions_h3_confu = bootstrap(transitions_h3)
        ydata = np.array([transitions_h1_mean, transitions_h2_mean, transitions_h3_mean])
        yerr = np.array([(transitions_h1_confb, transitions_h1_confu),
                         (transitions_h2_confb, transitions_h2_confu),
                         (transitions_h3_confb, transitions_h3_confu)])
        yerr = np.abs(yerr - ydata[:, None])
        rects2 = ax.bar(x + width / 2, ydata,
                        yerr=yerr.T, width=width, label=secondary_label,
                        color=label_to_color("vlgreen"), align='center', ecolor='black', capsize=10)
    labels = ['Human 1', 'Human 2', 'Model']
    ax.set_xticks(x)
    # ax.set_yticks(np.arange(0, 1.2, step=0.2))
    ax.set_xticklabels(labels)  # , rotation=-45
    ax.bar_label(rects1, fmt='%.2f', padding=3, label_type="center")
    if multi_dataset:
        ax.bar_label(rects2, fmt='%.2f', padding=3, label_type="center")
    ax.legend(loc='lower right')
    ax.set_ylabel("Transitions per 100 frames")
    save_path = args.output_folder
    if multi_dataset:
        name = "multi_dataset_transitions_per_100_bar_plot.pdf"
    else:
        name = "transitions_per_100_bar_plot.pdf"
    plt.savefig(str(Path(save_path, name)), bbox_inches='tight')
    plt.cla()
    plt.clf()
    plt.close(fig)


def generate_dataset_plots(sorted_IDs, all_metrics, args):
    """
    creates all the plots that relate to the entire dataset
    :param sorted_IDs:
    :param all_metrics:
    :param save_path:
    :return:
    """
    save_path = args.output_folder
    generate_barplot(sorted_IDs, all_metrics, save_path)
    generate_confusion_matrices(sorted_IDs, all_metrics, args)
    if args.raw_dataset_type == "lookit":
        generate_in_out_trial_vs_agreement(sorted_IDs, all_metrics, args)
        generate_posture_vs_agreement(sorted_IDs, all_metrics, args)
        generate_agreement_scatter(sorted_IDs, all_metrics, args, True)
        generate_confidence_vs_agreement(sorted_IDs, all_metrics, args, True)
        generate_transitions_plot(sorted_IDs, all_metrics, args, True)
        csv_file = Path(args.raw_dataset_path / args.db_file_name)
        video_dataset = preprocess.build_lookit_video_dataset(args.raw_dataset_path, csv_file)
        generate_categorial_vs_agreement(sorted_IDs, all_metrics, args, video_dataset, "child_eye_color")
        generate_categorial_vs_agreement(sorted_IDs, all_metrics, args, video_dataset, "child_skin_tone")
        generate_categorial_vs_agreement(sorted_IDs, all_metrics, args, video_dataset, "camera_moved")
        generate_categorial_vs_agreement(sorted_IDs, all_metrics, args, video_dataset, "child_race")
        generate_categorial_vs_agreement(sorted_IDs, all_metrics, args, video_dataset, "child_gender")
        generate_age_vs_agreement(sorted_IDs, all_metrics, args, video_dataset)
    elif args.raw_dataset_type == "cali-bw":
        generate_agreement_scatter(sorted_IDs, all_metrics, args, False)
        generate_confidence_vs_agreement(sorted_IDs, all_metrics, args, False)
        generate_transitions_plot(sorted_IDs, all_metrics, args, False)
        # csv_file = Path(args.raw_dataset_path / args.db_file_name)
        video_dataset = preprocess.build_marchman_video_dataset(args.raw_dataset_path, args.raw_dataset_type)
        generate_categorial_vs_agreement(sorted_IDs, all_metrics, args, video_dataset, "child_preterm")
        generate_categorial_vs_agreement(sorted_IDs, all_metrics, args, video_dataset, "child_race")
        generate_categorial_vs_agreement(sorted_IDs, all_metrics, args, video_dataset, "child_gender")
        generate_age_vs_agreement(sorted_IDs, all_metrics, args, video_dataset)
    elif args.raw_dataset_type == "senegal":
        generate_agreement_scatter(sorted_IDs, all_metrics, args, False)
        generate_confidence_vs_agreement(sorted_IDs, all_metrics, args, False)
        generate_transitions_plot(sorted_IDs, all_metrics, args, False)
        video_dataset = preprocess.build_marchman_video_dataset(args.raw_dataset_path, args.raw_dataset_type)
        generate_categorial_vs_agreement(sorted_IDs, all_metrics, args, video_dataset, "child_gender")
        generate_age_vs_agreement(sorted_IDs, all_metrics, args, video_dataset)
    elif args.raw_dataset_type == "just_annotations":
        # generate_in_out_trial_vs_agreement(sorted_IDs, all_metrics, args)
        # generate_posture_vs_agreement(sorted_IDs, all_metrics, args)
        generate_agreement_scatter(sorted_IDs, all_metrics, args, False)
        generate_confidence_vs_agreement(sorted_IDs, all_metrics, args, False)
        generate_transitions_plot(sorted_IDs, all_metrics, args, False)
    else:
        raise NotImplementedError


def generate_collage_plot(sorted_IDs, all_metrics, save_path):
    """
    plots one image with various selected stats
    :param sorted_IDs: ids of videos sorted by accuracy score
    :param all_metrics: all metrics per video
    :param save_path: where to save the image
    :return:
    """
    classes = {"away": 0, "left": 1, "right": 2}
    # fig, axs = plt.subplots(3, 2, figsize=(10, 12))
    fig = plt.figure(figsize=(15, 18))

    # accuracies plot
    accuracy_bar = fig.add_subplot(3, 2, (1, 2))  # three rows, two columns
    # accuracy_bar = axs[0, :]
    accuracies_hvh = [all_metrics[ID]["human1_vs_human2_session"]['agreement']*100 for ID in sorted_IDs]
    mean_hvh = np.mean(accuracies_hvh)
    accuracies_hvm = [all_metrics[ID]["human1_vs_machine_session"]['agreement']*100 for ID in sorted_IDs]
    mean_hvm = np.mean(accuracies_hvm)
    labels = sorted_IDs
    width = 0.35  # the width of the bars
    x = np.arange(len(labels))
    accuracy_bar.bar(x - width / 2, accuracies_hvh, width, color=label_to_color("lorange"), label='Human vs Human')
    accuracy_bar.bar(x + width / 2, accuracies_hvm, width, color=label_to_color("mblue"), label='Human vs Model')
    accuracy_bar.set_ylabel('Percent Agreement')
    accuracy_bar.set_xlabel('Video')
    accuracy_bar.set_title('Percent agreement per video')
    accuracy_bar.set_xticks(x)
    accuracy_bar.axhline(y=mean_hvh, color=label_to_color("lorange"), linestyle='-', label="mean (" + str(mean_hvh)[:4] + ")")
    accuracy_bar.axhline(y=mean_hvm, color=label_to_color("mblue"), linestyle='-', label="mean (" + str(mean_hvm)[:4] + ")")
    accuracy_bar.set_ylim([0, 100])
    accuracy_bar.legend()

    # target valid plot
    transitions_bar = fig.add_subplot(3, 2, 3)  # three rows, two columns
    width = 0.66  # the width of the bars
    x = np.arange(len(sorted_IDs))
    transitions_h1 = [100*all_metrics[ID]["human1_vs_human2_session"]['n_transitions_1'] /
                          all_metrics[ID]["human1_vs_human2_session"]['valid_frames_1'] for ID in sorted_IDs]
    transitions_h2 = [100*all_metrics[ID]["human1_vs_human2_session"]['n_transitions_2'] /
                          all_metrics[ID]["human1_vs_human2_session"]['valid_frames_2'] for ID in sorted_IDs]
    transitions_m = [100*all_metrics[ID]["human1_vs_machine_session"]['n_transitions_2'] /
                          all_metrics[ID]["human1_vs_machine_session"]['valid_frames_2'] for ID in sorted_IDs]

    transitions_bar.bar(x - width / 3, transitions_h1, width=(width / 3) - 0.1, label="Human 1", color=label_to_color("lorange"))
    transitions_bar.bar(x, transitions_h2, width=(width / 3) - 0.1, label="Human 2", color=label_to_color("lgreen"))
    transitions_bar.bar(x + width / 3, transitions_m, width=(width / 3) - 0.1, label="Model", color=label_to_color("mblue"))
    transitions_bar.set_xticks(x)
    transitions_bar.set_title('# Transitions per 100 frames')
    transitions_bar.legend()
    transitions_bar.set_ylabel('# Transitions per 100 frames')
    transitions_bar.set_xlabel('Video')

    # Looking time plot
    on_away_scatter = fig.add_subplot(3, 2, 4)  # three rows, two columns
    # on_away_scatter = axs[1, 1]
    on_away_scatter.plot([0, 1], [0, 1], transform=on_away_scatter.transAxes, color="black", label="Ideal trend")
    x_target_away_hvh = [all_metrics[ID]["human1_vs_human2_session"]['looking_time_1']/30 for ID in sorted_IDs]
    y_target_away_hvh = [all_metrics[ID]["human1_vs_human2_session"]['looking_time_2']/30 for ID in sorted_IDs]
    x_target_away_hvm = [all_metrics[ID]["human1_vs_machine_session"]['looking_time_1']/30 for ID in sorted_IDs]
    y_target_away_hvm = [all_metrics[ID]["human1_vs_machine_session"]['looking_time_2']/30 for ID in sorted_IDs]
    maxi = np.max(x_target_away_hvh + y_target_away_hvh + x_target_away_hvm + y_target_away_hvm)
    on_away_scatter.set_xlim([0, maxi])
    on_away_scatter.set_ylim([0, maxi])
    on_away_scatter.scatter(x_target_away_hvh, y_target_away_hvh, color=label_to_color("lorange"), label='Human vs Human')
    # for i in range(len(sorted_IDs)):
    #     on_away_scatter.annotate(i, (x_target_away_hvh[i], y_target_away_hvh[i]))
    on_away_scatter.scatter(x_target_away_hvm, y_target_away_hvm, color=label_to_color("mblue"), label='Human vs Model')
    # for i in range(len(sorted_IDs)):
    #     on_away_scatter.annotate(i, (x_target_away_hvm[i], y_target_away_hvm[i]))
    on_away_scatter.set_xlabel("Human 1")
    on_away_scatter.set_ylabel("Human 2 or Model")
    on_away_scatter.set_title("Looking time [s]")
    on_away_scatter.legend()

    # label distribution plot
    # label_scatter = fig.add_subplot(3, 2, 5)  # three rows, two columns
    # # label_scatter = axs[2, 0]
    # label_scatter.plot([0, 1], [0, 1], transform=label_scatter.transAxes, color="black", label="Ideal trend")
    # label_scatter.set_xlim([0, 1])
    # label_scatter.set_ylim([0, 1])
    # for i, label in enumerate(sorted(classes.keys())):
    #     y_labels = [y[i] / sum(y) for y in [all_metrics[ID]["human1_vs_human2"]['coding1_by_label'] for ID in sorted_IDs]]
    #     x_labels = [y[i] / sum(y) for y in [all_metrics[ID]["human1_vs_human2"]['coding2_by_label'] for ID in sorted_IDs]]
    #     label_scatter.scatter(x_labels, y_labels, color=label_to_color(label), label="hvh: " + label, marker='^')
    #     for n in range(len(sorted_IDs)):
    #         label_scatter.annotate(n, (x_labels[n], y_labels[n]))
    # for i, label in enumerate(sorted(classes.keys())):
    #     y_labels = [y[i] / sum(y) for y in [all_metrics[ID]["human1_vs_machine"]['coding1_by_label'] for ID in sorted_IDs]]
    #     x_labels = [y[i] / sum(y) for y in [all_metrics[ID]["human1_vs_machine"]['coding2_by_label'] for ID in sorted_IDs]]
    #     label_scatter.scatter(x_labels, y_labels, color=label_to_color(label), label="hvh: " + label, marker='o')
    #     for n in range(len(sorted_IDs)):
    #         label_scatter.annotate(n, (x_labels[n], y_labels[n]))
    #
    # label_scatter.set_xlabel('Human 1 label proportion')
    # label_scatter.set_ylabel('Human 2 / Machine labels proportion')
    # label_scatter.set_title('labels distribution')
    # label_scatter.legend()  # loc='upper center'

    # label distribution bar plot
    label_bar = fig.add_subplot(3, 2, (5, 6))  # three rows, two columns
    # label_bar = axs[2, 1]
    ticks = range(len(sorted_IDs))
    bottoms_h1 = np.zeros(shape=(len(sorted_IDs)))
    bottoms_h2 = np.zeros(shape=(len(sorted_IDs)))
    bottoms_m = np.zeros(shape=(len(sorted_IDs)))
    width = 0.66
    patterns = [None, "O", "*"]
    for i, label in enumerate(sorted(classes.keys())):
        label_counts_h1 = [y[i] / sum(y[:3]) for y in [all_metrics[ID]["human1_vs_human2_session"]['label_count_1'] for ID in sorted_IDs]]
        label_counts_h2 = [y[i] / sum(y[:3]) for y in [all_metrics[ID]["human1_vs_human2_session"]['label_count_2'] for ID in sorted_IDs]]
        label_counts_m = [y[i] / sum(y[:3]) for y in [all_metrics[ID]["human1_vs_machine_session"]['label_count_2'] for ID in sorted_IDs]]

        label_bar.bar(x - width/3, label_counts_h1, bottom=bottoms_h1, width=(width / 3)-0.1, label=label,
                      color=label_to_color(label), edgecolor='black', hatch=patterns[0], linewidth=0)
        label_bar.bar(x, label_counts_h2, bottom=bottoms_h2, width=(width / 3)-0.1, label=label,
                      color=label_to_color(label), edgecolor='black', hatch=patterns[1], linewidth=0)
        label_bar.bar(x + width/3, label_counts_m, bottom=bottoms_m, width=(width / 3)-0.1, label=label,
                      color=label_to_color(label), edgecolor='black', hatch=patterns[2], linewidth=0)
        if i == 0:
            artists = [Patch(facecolor=label_to_color("away"), label="Away"),
                       Patch(facecolor=label_to_color("left"), label="Left"),
                       Patch(facecolor=label_to_color("right"), label="Right"),
                       Patch(facecolor="white", edgecolor='black', hatch=patterns[0], label="Human 1"),
                       Patch(facecolor="white", edgecolor='black', hatch=patterns[1], label="Human 2"),
                       Patch(facecolor="white", edgecolor='black', hatch=patterns[2], label="Model")]
            label_bar.legend(handles=artists, bbox_to_anchor=(0.95, 1.0), loc='upper left')
        bottoms_h1 += label_counts_h1
        bottoms_h2 += label_counts_h2
        bottoms_m += label_counts_m
    label_bar.xaxis.set_major_locator(MultipleLocator(1))
    label_bar.set_xticks(ticks)
    label_bar.set_title('Proportion of looking left, right, and away per video')
    label_bar.set_ylabel('Proportion')
    label_bar.set_xlabel('Video')

    plt.subplots_adjust(left=0.1, bottom=0.075, right=0.9, top=0.925, wspace=0.2, hspace=0.5)
    plt.savefig(Path(save_path, "collage.png"))
    plt.cla()
    plt.clf()
    plt.close(fig)


def plot_luminance_vs_accuracy(sorted_IDs, all_metrics, args, hvh=False):
    plt.rc('font', size=16)
    fig, ax = plt.subplots()
    # plt.figure(figsize=(8.0, 6.0))
    plt_name = "agreement_vs_luminance"
    if hvh:
        session_metric_name = "human1_vs_human2_session"
        plt_name += "_humans"
    else:
        session_metric_name = "human1_vs_machine_session"
        plt_name += "_machine"
    plt_name += ".pdf"
    if args.raw_dataset_type == "cali-bw":
        color = label_to_color("vlgreen")
    elif args.raw_dataset_type == "senegal":
        color = label_to_color("vlpurple")
    else:
        color = label_to_color("vlblue")
    x = []
    for i, id in enumerate(sorted_IDs):
        x.append(all_metrics[id]["stats"]["luminance"])
    y = [all_metrics[id][session_metric_name]["agreement"] for id in sorted_IDs]
    sns.regplot(x=x, y=y, color=color)
    ax.set_xlabel("Luminance")
    ax.set_ylabel("Percent Agreement")
    save_path = args.output_folder
    plt.savefig(str(Path(save_path, plt_name)), bbox_inches='tight')
    plt.cla()
    plt.clf()
    plt.close(fig)


def get_face_stats(id, faces_folder, start=0, end=None, mask=None):
    """
    given a video id, calculates the average face area in pixels using pre-processed crops
    :param ids: video id
    :param faces_folder: the folder containing all crops and their meta data as created by "preprocess.py"
    :param start: if provided, slices the faces according to this start point
    :param end: if provided, slices the faces according to this end point
    :param mask: if provided, slices the faces according to these indices (must be np array of same size)
    :return:
    """
    face_areas = []
    face_locs = []
    if faces_folder is None:
        return None
    file = Path(faces_folder, id, 'face_labels_fc.npy')
    if file.is_file():
        face_labels = np.load(file)
        if end is not None:
            face_labels = face_labels[start:end]
        for i, face_id in enumerate(face_labels):
            if mask is not None:
                if not mask[i]:
                    continue
            if face_id >= 0:
                box_file = Path(faces_folder, id, "box", "{:05d}_{:01d}.npy".format(i+start, face_id))
                '{name}/box/{frame_number + i:05d}_{face_label_seg[i]:01d}.npy.format()'
                box = np.load(box_file, allow_pickle=True).item()
                # face box represents (top, bottom, left, right) of bbox respectively (top has lower value than bottom)
                face_area = (box['face_box'][1] - box['face_box'][0]) * (box['face_box'][3] - box['face_box'][2])  # y * x
                img_shape = box["img_shape"]  # y, x
                face_loc = np.array([(box['face_box'][3] + box['face_box'][2]) / 2,
                                     (box['face_box'][1] + box['face_box'][0]) / 2])  # x, y (y grows downwards)
                face_loc[0] /= img_shape[1]
                face_loc[1] /= img_shape[0]
                face_loc[1] = 1 - face_loc[1]  # flip y to grow upwards
                face_areas.append(face_area)
                face_locs.append(face_loc)
        if len(face_areas) > 0:
            face_areas = np.mean(face_areas)
        else:
            face_areas = np.nan
        if len(face_locs) > 0:
            face_locs = np.mean(face_locs, axis=0)
            face_stds = np.mean(np.std(face_locs, axis=0))
        else:
            face_locs = np.array([np.nan, np.nan])
            face_stds = np.nan
        return face_areas, face_locs, face_stds
    else:
        return None


def plot_face_pixel_density_vs_accuracy(sorted_IDs, all_metrics, args, trial_level=False, hvh=False):
    plt.rc('font', size=16)
    # fig, ax = plt.subplots()
    plt_name = "agreement_vs_face_density"
    if hvh:
        trial_metric_name = "human1_vs_human2_trials"
        session_metric_name = "human1_vs_human2_session"
        plt_name += "_humans"
    else:
        trial_metric_name = "human1_vs_machine_trials"
        session_metric_name = "human1_vs_machine_session"
        plt_name += "_machine"
    if trial_level:
        plt_name += "_trials.pdf"
        densities = []
        agreement = []
        for ID in sorted_ids:
            densities += [x["avg_face_pixel_density"] for x in all_metrics[ID]["human1_vs_machine_trials"]]
            agreement += [x["agreement"] for x in all_metrics[ID][trial_metric_name]]
        densities = np.array(densities)
        agreement = np.array(agreement)
        alpha = 0.3
    else:
        plt_name += "_sessions.pdf"
        densities = [all_metrics[x]["stats"]["avg_face_pixel_density"] for x in sorted_IDs]
        agreement = [all_metrics[id][session_metric_name]["agreement"] for id in sorted_IDs]
        alpha = 1
    if args.raw_dataset_type == "cali-bw":
        color = label_to_color("vlgreen")
    elif args.raw_dataset_type == "senegal":
        color = label_to_color("vlpurple")
    else:
        color = label_to_color("vlblue")

    ax = sns.regplot(x=densities, y=agreement, color=color, scatter_kws={"alpha": alpha})
    ax.set_xlabel("Face pixel density")
    # ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
    # ax.ticklabel_format(axis="x", style="sci")
    ax.set_ylabel("Percent Agreement")
    ax.ticklabel_format(style='sci', axis='x', scilimits=[-5, 4])
    save_path = args.output_folder
    plt.savefig(str(Path(save_path, plt_name)), bbox_inches='tight')
    plt.cla()
    plt.clf()
    # plt.close(fig)


def plot_face_location_vs_accuracy(sorted_IDs, all_metrics, args, use_x=True, trial_level=False, hvh=False):
    plt.rc('font', size=16)
    fig, ax = plt.subplots()
    if use_x:
        stat = 0
        x_label = "Face Horizontal Position"
        plt_name = "agreement_vs_face_locx"
    else:
        stat = 1
        x_label = "Face Vertical Position"
        plt_name = "agreement_vs_face_locy"
    if hvh:
        trial_metric_name = "human1_vs_human2_trials"
        session_metric_name = "human1_vs_human2_session"
        plt_name +="_humans"
    else:
        trial_metric_name = "human1_vs_machine_trials"
        session_metric_name = "human1_vs_machine_session"
        plt_name += "_machine"
    if trial_level:
        plt_name += "_trials.pdf"
        means = []
        agreement = []
        for ID in sorted_ids:
            means += [x["avg_face_loc"] for x in all_metrics[ID]["human1_vs_machine_trials"]]
            agreement += [x["agreement"] for x in all_metrics[ID][trial_metric_name]]
        # create np array dtype=float64 with nans where original means had nan
        # means = np.array([np.array([np.nan, np.nan]) if np.any(np.isnan(x)) else x for x in means]).squeeze()
        means = np.array(means)[:, stat]
        agreement = np.array(agreement)
        alpha = 0.3
    else:
        plt_name += "_sessions.pdf"
        means = np.array([all_metrics[x]["stats"]["avg_face_loc"][stat] for x in sorted_IDs])
        agreement = np.array([all_metrics[id][session_metric_name]["agreement"] for id in sorted_IDs])
        alpha = 1
    if args.raw_dataset_type == "cali-bw":
        color = label_to_color("vlgreen")
    elif args.raw_dataset_type == "senegal":
        color = label_to_color("vlpurple")
    else:
        color = label_to_color("vlblue")
    means = means - 0.5  # move from 0:1 to -0.5:0.5
    sns.regplot(x=means, y=agreement, color=color, scatter_kws={"alpha": alpha})
    ax.set_xlabel(x_label)
    ax.set_ylabel("Percent Agreement")
    ax.set_xlim([-0.5, 0.5])
    save_path = args.output_folder
    plt.savefig(str(Path(save_path, plt_name)), bbox_inches='tight')
    plt.cla()
    plt.clf()
    plt.close(fig)


def plot_face_location_std_vs_accuracy(sorted_IDs, all_metrics, args, trial_level=False, hvh=False):
    plt.rc('font', size=16)
    fig, ax = plt.subplots()
    plt_name = "agreement_vs_face_loc_std"
    if hvh:
        trial_metric_name = "human1_vs_human2_trials"
        session_metric_name = "human1_vs_human2_session"
        plt_name += "_humans"
    else:
        trial_metric_name = "human1_vs_machine_trials"
        session_metric_name = "human1_vs_machine_session"
        plt_name += "_machine"
    if trial_level:
        plt_name += "_trials.pdf"
        stds = []
        agreement = []
        for ID in sorted_ids:
            stds += [x["avg_face_loc_std"] for x in all_metrics[ID]["human1_vs_machine_trials"]]
            agreement += [x["agreement"] for x in all_metrics[ID][trial_metric_name]]
        stds = np.array(stds)
        agreement = np.array(agreement)
        alpha = 0.3
    else:
        plt_name += "_sessions.pdf"
        stds = [all_metrics[x]["stats"]["avg_face_loc_std"] for x in sorted_IDs]
        agreement = [all_metrics[id][session_metric_name]["agreement"] for id in sorted_IDs]
        alpha = 1
    if args.raw_dataset_type == "cali-bw":
        color = label_to_color("vlgreen")
    elif args.raw_dataset_type == "senegal":
        color = label_to_color("vlpurple")
    else:
        color = label_to_color("vlblue")
    sns.regplot(x=stds, y=agreement, color=color, scatter_kws={"alpha": alpha})
    ax.set_xlabel("Face location std in pixels")
    ax.set_ylabel("Percent Agreement")
    save_path = args.output_folder
    plt.savefig(str(Path(save_path, plt_name)), bbox_inches='tight')
    plt.cla()
    plt.clf()
    plt.close(fig)


def calc_all_metrics(args, force_create=False):
    """
    calculates all relevant metrics and stores result in file metrics.p
    :return:
    """
    metric_save_path = Path(args.output_folder, "metrics.p")
    if metric_save_path.is_file() and not force_create:
        all_metrics = pickle.load(open(metric_save_path, "rb"))
    else:
        machine_annotation = []
        human_annotation = []
        human_annotation2 = []

        # Get a list of all machine annotation files
        for file in Path(args.human_codings_folder).glob("*"):
            human_annotation.append(file.stem)
            human_ext = file.suffix
        for file in Path(args.machine_codings_folder).glob("*"):
            machine_annotation.append(file.stem)
            machine_ext = file.suffix
        # Intersect the sets to get a list of mutually coded videos
        coding_intersect = set(machine_annotation).intersection(set(human_annotation))
        if args.human2_codings_folder:
            for file in Path(args.human2_codings_folder).glob("*"):
                human_annotation2.append(file.stem)
                human2_ext = file.suffix
            coding_intersect = coding_intersect.intersection(set(human_annotation2))

        if args.raw_dataset_type == "lookit":
            video_dataset = preprocess.build_lookit_video_dataset(args.raw_dataset_path,
                                                                  Path(args.raw_dataset_path, args.db_file_name))
        elif args.raw_dataset_type == "cali-bw" or args.raw_dataset_type == "senegal":
            video_dataset = preprocess.build_marchman_video_dataset(args.raw_dataset_path,
                                                                    args.raw_dataset_type)
        else:
            video_dataset = None
        
        if args.unique_children_only and video_dataset is not None:  # allow only one video per child
            filter_files = [x for x in video_dataset.values() if
                            x["in_csv"] and x["has_1coding"] and x["has_2coding"] and x[
                                "split"] == "2_test"]
            video_children_id = [x["child_id"] for x in filter_files]
            _, indices = np.unique(video_children_id, return_index=True)
            unique_videos = np.array(filter_files)[indices].tolist()
            unique_videos = [x["video_id"] for x in unique_videos]
            coding_intersect = coding_intersect.intersection(set(unique_videos))

        # sort the file paths alphabetically to pair them up
        coding_intersect = sorted(list(coding_intersect))

        assert len(coding_intersect) > 0
        all_metrics = {}
        for i, code_file in enumerate(coding_intersect):
            logging.info("{} / {} / computing stats for {}".format(i, len(coding_intersect) - 1, code_file))
            human_coding_file = Path(args.human_codings_folder, code_file + human_ext)
            machine_coding_file = Path(args.machine_codings_folder, code_file + machine_ext)
            if args.human2_codings_folder:
                human_coding_file2 = Path(args.human2_codings_folder, code_file + human2_ext)
            else:
                human_coding_file2 = None
            key = human_coding_file.stem
            try:
                # hack for marchman style datasets because we accidentally mapped human1 to reliability coder
                if args.raw_dataset_type == "cali-bw" or args.raw_dataset_type == "senegal":
                    res = compare_coding_files(human_coding_file2, human_coding_file, machine_coding_file, args)
                else:
                    res = compare_coding_files(human_coding_file, human_coding_file2, machine_coding_file, args)
            except (IndexError, AssertionError) as e:
                logging.warning("skipped: {}, because of failure:".format(key))
                logging.warning(e)
                continue
            all_metrics[key] = res
            # other stats
            if args.faces_folder is not None:
                face_stats = get_face_stats(key, args.faces_folder)
                all_metrics[key]["stats"]["avg_face_pixel_density"] = face_stats[0]
                all_metrics[key]["stats"]["avg_face_loc"] = face_stats[1]  # x, y
                all_metrics[key]["stats"]["avg_face_loc_std"] = np.mean(face_stats[2])
                if args.raw_video_folder is not None:
                    all_metrics[key]["stats"]["luminance"] = sample_luminance(key, args.raw_video_folder,
                                                                            all_metrics[key]["human1_vs_machine_session"]['start'],
                                                                            all_metrics[key]["human1_vs_machine_session"]['end'])
            if video_dataset is not None:
                all_metrics[key]["csv_info"] = video_dataset[key]  # add participant info just in case
        # Store in disk for faster access next time:
        pickle.dump(all_metrics, open(metric_save_path, "wb"))
    return all_metrics


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


def make_gridview(array, ncols=3, save_path=None):
    """
    grid views of a set of images
    :param array: the images n x H x W x 3 (must have n = x^2 where x is int)
    :param ncols: the number of columns inthe grid view
    :param save_path: the path to save the gridview
    :return: the gridview np array (nrows x ncols x H x W x 3)
    """
    nindex, height, width, intensity = array.shape
    nrows = nindex//ncols
    assert nindex == nrows*ncols
    result = (array.reshape(nrows, ncols, height, width, intensity)
              .swapaxes(1,2)
              .reshape(height*nrows, width*ncols, intensity))
    if save_path is not None:
        plt.imshow(result)
        plt.savefig(save_path)
        plt.cla()
        plt.clf()
    return result


def prep_frame(frame, bbox, show_bbox=True, show_arrow=False, conf=None, class_text=None, rect_color=None, frame_number=None, pic_in_pic=False):
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
        if class_text is not None:
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


def temp_hook(frame, cv2_bboxes, frame_counter):
    """
    temporary function to create illustrations for manuscript/github repository
    :param frame:
    :param cv2_bboxes:
    :param frame_counter:
    :return:
    """
    save_path = Path("./plots/", "for_draft")

    fig, ax = plt.subplots()
    ax.imshow(frame)
    ax.set_axis_off()
    fig.tight_layout()
    plt.savefig(str(Path(save_path, "clean_frame_gallery_{:04d}.png".format(frame_counter))))
    plt.cla()
    plt.clf()
    plt.close(fig)

    for i, bbox in enumerate(cv2_bboxes):
        crop = frame[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]
        crop = cv2.resize(crop, (100, 100))
        fig, ax = plt.subplots()
        ax.imshow(crop)
        ax.set_axis_off()
        fig.tight_layout()
        save_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(Path(save_path, "{}_crop_{:04d}.png".format(i, frame_counter))))
        plt.cla()
        plt.clf()
        plt.close(fig)

    box = cv2_bboxes.pop(1)
    cv2_bboxes.append(box)
    for i, box in enumerate(cv2_bboxes):
        fig, ax = plt.subplots()
        new_frame = prep_frame(frame, box)
        ax.imshow(new_frame)
        ax.set_axis_off()
        fig.tight_layout()
        save_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(Path(save_path, "{}_bbox_frame_gallery_{:04d}.png".format(i, frame_counter))))
        plt.cla()
        plt.clf()
        plt.close(fig)


def print_stats(sorted_ids, all_metrics, hvm, args):
    """
    prints a bunch of metrics for H1 vs M and H1 vs H2
    :param sorted_ids: the ids of the videos
    :param all_metrics: all possible metrics
    :param hvm: whether to print metrics for H1 v M or H1 v H2
    :param args: command line arguments
    :return:
    """
    if hvm:
        choice1 = "human1_vs_machine_session"
        choice2 = "ICC_LT_hvm"
        choice3 = "ICC_PR_hvm"
    else:
        choice1 = "human1_vs_human2_session"
        choice2 = "ICC_LT_hvh"
        choice3 = "ICC_PR_hvh"
    agreement = [all_metrics[ID][choice1]["agreement"] for ID in sorted_ids]
    kappa = [all_metrics[ID][choice1]["kappa"] for ID in sorted_ids]
    invalid = [1 - (all_metrics[ID][choice1]["valid_frames_2"] /
                    all_metrics[ID][choice1]["n_frames_in_interval"]) for ID in sorted_ids]
    ICC_LT = [all_metrics[ID]["stats"][choice2] for ID in sorted_ids]
    ICC_PR = [all_metrics[ID]["stats"][choice3] for ID in sorted_ids]
    invalid_mean, invalid_conf1, invalid_conf2 = bootstrap(np.array(invalid)*100)
    # invalid_mean = np.mean(invalid) * 100
    # invalid_std = np.std(invalid) * 100
    agreement_mean, agreement_conf1, agreement_conf2 = bootstrap(np.array(agreement)*100)
    # agreement_mean = np.mean(agreement) * 100
    # agreement_std = np.std(agreement) * 100
    kappa_mean, kappa_conf1, kappa_conf2 = bootstrap(kappa)
    # kappa_mean = np.mean(kappa)
    # kappa_std = np.std(kappa)
    ICC_LT_mean, ICC_LT_conf1, ICC_LT_conf2 = bootstrap(ICC_LT)
    # ICC_LT_mean = np.mean(ICC_LT)
    # ICC_LT_std = np.std(ICC_LT)
    if args.raw_dataset_type == "datavyu":
        ICC_PR_mean, ICC_PR_conf1, ICC_PR_conf2 = 0, 0, 0
    else:
        ICC_PR_mean, ICC_PR_conf1, ICC_PR_conf2 = bootstrap(ICC_PR)
    # ICC_PR_mean = np.mean(ICC_PR)
    # ICC_PR_std = np.std(ICC_PR)

    if args.raw_dataset_type == "lookit":
        data = np.load("cali-bw_agreement.npz")
        cali_hvh, cali_hvm = data["arr_0"], data["arr_1"]

    print("hvm: {}".format(hvm))
    if hvm:
        if args.raw_dataset_type == "lookit":
            t, p, dof = t_test(cali_hvm, agreement)
            print("t-test (unpaired) hvm agreement: t={:.2f}, p={:.8f}, dof={:.2f}]".format(t, p, dof))
        t, p, dof = t_test_paired(ICC_LT, ICC_PR)
        print("t-test (paired) hvm LT vs PR: t={:.2f}, p={:.8f}, dof={}]".format(t, p, dof))
        confidence_correct = []
        confidence_incorrect = []
        for ID in sorted_ids:
            confidence_correct += [x["confidence_metrics"][0] for x in all_metrics[ID]["human1_vs_machine_trials"]]
            confidence_incorrect += [x["confidence_metrics"][1] for x in all_metrics[ID]["human1_vs_machine_trials"]]
        confidence_correct = np.array(confidence_correct)
        confidence_incorrect = np.array(confidence_incorrect)
        valid_trials_confidence = ~np.isnan(confidence_correct) & ~np.isnan(confidence_incorrect)
        confidence_correct = confidence_correct[valid_trials_confidence]
        confidence_incorrect = confidence_incorrect[valid_trials_confidence]
        t, p, dof = t_test(confidence_correct, confidence_incorrect)
        print("t-test (unpaired) hvm confidence: t={:.2f}, p={:.8f}, dof={:.2f}".format(t, p, dof))
        if args.raw_dataset_type != "datavyu":
            transitions_h1 = [100 * all_metrics[ID]["human1_vs_machine_session"]['n_transitions_1'] /
                              all_metrics[ID]["human1_vs_human2_session"]['valid_frames_1'] for ID in sorted_ids]
            transitions_h2 = [100 * all_metrics[ID]["human1_vs_machine_session"]['n_transitions_2'] /
                              all_metrics[ID]["human1_vs_machine_session"]['valid_frames_2'] for ID in sorted_ids]
            transitions_h1 = np.array(transitions_h1)
            transitions_h2 = np.array(transitions_h2)
            t, p, dof = t_test_paired(transitions_h1, transitions_h2)
            print("t-test (paired) hvm transitions: t={:.2f}, p={:.8f}, dof={}".format(t, p, dof))
        invalid_no_face = 0
        invalid_no_infant_face = 0
        invalid_unable_to_predict = 0
        invalid_n = 0
        for ID in sorted_ids:
            raw2 = all_metrics[ID][choice1]["raw_coding2"]
            invalids = raw2[raw2 < 0]
            invalid_no_face += np.count_nonzero(invalids == -2)
            invalid_no_infant_face += np.count_nonzero(invalids == -1)
            invalid_unable_to_predict += np.count_nonzero(invalids == -3)
            invalid_n += len(invalids)
        invalid_no_face = 100 * invalid_no_face / invalid_n
        invalid_no_infant_face = 100 * invalid_no_infant_face / invalid_n
        invalid_unable_to_predict = 100 * invalid_unable_to_predict / invalid_n

        print("breakdown of invalids: NO_FACE: {:.2f}%, NO_INFANT_FACE: {:.2f}%, UNABLE_TO_PREDICT: {:.2f}%".format(
            invalid_no_face,
            invalid_no_infant_face,
            invalid_unable_to_predict))
    else:  # hvh
        if args.raw_dataset_type == "lookit":
            t, p, dof = t_test(cali_hvh, agreement)
            print("t-test (unpaired) hvh agreement: t={:.2f}, p={:.8f}, dof={:.2f}".format(t, p, dof))
        t, p, dof = t_test_paired(ICC_LT, ICC_PR)
        print("t-test (paired) hvh LT vs PR: t={:.2f}, p={:.8f}, dof={}]".format(t, p, dof))
        disagree_ratios = []
        for ID in sorted_ids:
            raw1 = all_metrics[ID]["human1_vs_human2_session"]["raw_coding1"]
            raw2 = all_metrics[ID]["human1_vs_human2_session"]["raw_coding2"]
            raw3 = all_metrics[ID]["human1_vs_machine_session"]["raw_coding2"]
            start = max(all_metrics[ID]["human1_vs_human2_session"]["start"],
                        all_metrics[ID]["human1_vs_machine_session"]["start"])
            end = min(all_metrics[ID]["human1_vs_human2_session"]["end"],
                        all_metrics[ID]["human1_vs_machine_session"]["end"])
            raw1 = raw1[start:end]
            raw2 = raw2[start:end]
            raw3 = raw3[start:end]
            mutually_valid_frames = np.logical_and(raw3, np.logical_and(raw1 >= 0, raw2 >= 0))
            humans_agree = raw1[mutually_valid_frames] == raw2[mutually_valid_frames]
            humans_disagree = ~humans_agree
            machine_disagree = raw1[mutually_valid_frames] != raw3[mutually_valid_frames]
            # machine_agree = ~machine_disagree
            total_disagree = np.sum(machine_disagree & humans_disagree)
            total_human_disagree = np.sum(humans_disagree)
            disagree_ratios.append(100 * total_disagree / total_human_disagree)
        disagree_ratio_mean, disagree_ratio_conf1, disagree_ratio_conf2 = bootstrap(np.array(disagree_ratios))
        print("disagree ratio: {:.2f}% [{:.2f}%, {:.2f}%]".format(disagree_ratio_mean,
                                                               disagree_ratio_conf1,
                                                               disagree_ratio_conf2))
    print("percent agreement: trial: {:.2f}% [{:.2f}%, {:.2f}%]".format(agreement_mean, agreement_conf1, agreement_conf2))
    print("% of invalid frames: {:.2f}% [{:.2f}%, {:.2f}%]".format(invalid_mean, invalid_conf1, invalid_conf2))
    print("Cohens Kappa: {:.2f} [{:.2f}, {:.2f}]".format(kappa_mean, kappa_conf1, kappa_conf2))
    print("ICC LT: {:.2f} [{:.2f}, {:.2f}]".format(ICC_LT_mean, ICC_LT_conf1, ICC_LT_conf2))
    print("ICC PR: {:.2f} [{:.2f}, {:.2f}]".format(ICC_PR_mean, ICC_PR_conf1, ICC_PR_conf2))


if __name__ == "__main__":
    args = parse_arguments_for_visualizations()
    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(filename=args.log, filemode='w', level=args.verbosity.upper())
    else:
        logging.basicConfig(level=args.verbosity.upper())
    all_metrics = calc_all_metrics(args, force_create=False)
    # sort by percent agreement
    sorted_ids = sorted(list(all_metrics.keys()),
                        key=lambda x: all_metrics[x]["human1_vs_machine_session"]["agreement"])
    print_stats(sorted_ids, all_metrics, True, args)

    if args.human2_codings_folder:
        print_stats(sorted_ids, all_metrics, False, args)
        generate_collage_plot(sorted_ids, all_metrics, args.output_folder)
        generate_collage_plot2(sorted_ids, all_metrics, args.output_folder)
        generate_dataset_plots(sorted_ids, all_metrics, args)
        if args.faces_folder:
            plot_face_pixel_density_vs_accuracy(sorted_ids, all_metrics, args)
            plot_face_pixel_density_vs_accuracy(sorted_ids, all_metrics, args, trial_level=True)
            plot_face_pixel_density_vs_accuracy(sorted_ids, all_metrics, args, trial_level=True, hvh=True)
            plot_face_location_std_vs_accuracy(sorted_ids, all_metrics, args)
            plot_face_location_std_vs_accuracy(sorted_ids, all_metrics, args, trial_level=True)
            plot_face_location_std_vs_accuracy(sorted_ids, all_metrics, args, trial_level=True, hvh=True)
            plot_face_location_vs_accuracy(sorted_ids, all_metrics, args)
            plot_face_location_vs_accuracy(sorted_ids, all_metrics, args, use_x=False)
            plot_face_location_vs_accuracy(sorted_ids, all_metrics, args, trial_level=True)
            plot_face_location_vs_accuracy(sorted_ids, all_metrics, args, trial_level=True, hvh=True)
            plot_face_location_vs_accuracy(sorted_ids, all_metrics, args, use_x=False, trial_level=True)
            plot_face_location_vs_accuracy(sorted_ids, all_metrics, args, use_x=False, trial_level=True, hvh=True)
            plot_luminance_vs_accuracy(sorted_ids, all_metrics, args)
            plot_luminance_vs_accuracy(sorted_ids, all_metrics, args, hvh=True)
        generate_session_plots(sorted_ids, all_metrics, args, anonymous=True)
