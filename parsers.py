import xml.etree.ElementTree as ET
import logging
from pathlib import Path
import numpy as np
import preprocess
import pandas as pd


class BaseParser:
    def __init__(self):
        self.classes = {'away': 0, 'left': 1, 'right': 2}

    def parse(self, video_id, label_path):
        """
        returns a list of lists. each list contains the frame number (or timestamps), valid_flag, class
        where:
        frame number is zero indexed (or if timestamp, starts from 0.0)
        valid_flag is 1 if this frame has valid annotation, and 0 otherwise
        class is either away, left, right or off.

        list should only contain frames that have changes in class (compared to previous frame)
        i.e. if the video is labeled ["away","away","away","right","right"]
        then only frame 0 and frame 3 will appear on the output list.

        :param video_id: the id of the video that label_path belongs to.
        :param label_path: the label file to parse.
        :return: None if failed, else: list of lists as described above, the frame which codings starts, and frame at which it ends
        """
        raise NotImplementedError

    def uncollapse_labels(self, labels, start, end, class_map=None):
        """
        given an output from parse as described above, uncollapses it into one big numpy array of labels (-3 for invalid).
        :param labels: the collapsed version of the labels
        :param start: index where coding begins
        :param end: index where coding ends
        :param class_map: if provided, uses this dictionary to map class names into integers
        :return:
        """
        if class_map is None:
            class_map = {"away": 0, "left": 1, "right": 2}
        if type(labels) == np.ndarray:
            return labels
        output = []
        for _ in range(start):
            output.append(-3)
        prev_entry = labels[0]
        for entry in labels[1:]:
            if prev_entry[1]:  # valid
                response = class_map[prev_entry[2]]
            else:
                response = -3
            for i in range(prev_entry[0], entry[0]):
                output.append(response)
            prev_entry = entry
        for i in range(end - len(output)):
            if labels[-1][1]:
                output.append(class_map[labels[-1][2]])
            else:
                output.append(-3)
        output = np.array(output)
        return output


class TrivialParser(BaseParser):
    """
    A trivial toy parser that labels all video as "left" if input "file" is not None
    """
    def __init__(self):
        super().__init__()

    def parse(self, video_id, label_path=None):
        if label_path:
            return [[0, 1, "left"]], 0, 10
        else:
            return None


class CompressedParser(BaseParser):
    """
    parses a npz file saved for visualizations
    to see how it is created check out test.py
    """
    def __init__(self):
        super().__init__()

    def parse(self, video_id, label_path=None):
        data = np.load(label_path)
        data = data["arr_0"]
        data[:4] = -3  # mark first frames as invalid
        data[-4:] = -3  # mark last frames as invalid
        return data, 4, len(data)-4

    def get_confidence(self, label_path):
        data = np.load(label_path)
        confidence = data["arr_1"]
        confidence[:4] = -1  # mark first frames as invalid
        confidence[-4:] = -1  # mark last frames as invalid
        return confidence


class LookitParser(BaseParser):
    """
    a parser that parses Lookit format, a slightly different version of PrefLookTimestampParser.
    """
    def __init__(self, fps, csv_file=None, first_coder=True, return_time_stamps=False):
        super().__init__()
        self.fps = fps
        self.return_time_stamps = return_time_stamps
        if csv_file is not None:
            self.video_dataset = preprocess.build_lookit_video_dataset(csv_file.parent, csv_file)
        else:
            self.video_dataset = None
        self.first_coder = first_coder
        self.classes = ["away", "left", "right"]
        self.exclude = ["outofframe", "preview", "instructions"]
        self.special = ["codingactive"]
        self.poses = ["over_shoulder", "sitting_in_lap", "sitting_alone", "other_posture", "no_posture"]

    def parse(self, video_id, label_path=None, extract_poses=False):
        """
        parse a coding file, see base class for output format
        :param video_id: video_id of video
        :param label_path: if provided, will parse this file instead
        :return: see base class
        """
        if extract_poses:
            selected_classes = self.poses
        else:
            selected_classes = self.classes
        if label_path is None:
            if self.video_dataset is None:
                raise ValueError("no label path provided and no csv file provided on initialization")
            else:
                if self.first_coder:
                    label_path = self.video_dataset[video_id]["first_coding_file"]
                else:
                    label_path = self.video_dataset[video_id]["second_coding_file"]
                if label_path is None:
                    logging.warning("Video ID: " + str(video_id) + " no matching vcx was found.")
                    return None
                if not label_path.is_file():
                    logging.warning("For the file: " + str(label_path) + " no matching vcx was found.")
                    return None
        labels = self.load_and_sort(label_path)
        # initialize
        output = []
        prev_class = "none"
        prev_frame = -1
        # loop over legitimate class labels
        for i in range(len(labels)):
            frame = int(labels[i, 0])
            if labels[i, 2] in selected_classes:
                cur_class = labels[i, 2]
                if prev_class != cur_class:
                    assert frame > prev_frame  # how can two labels be different but point to same time?
                output.append([frame, True, cur_class])
                prev_class = cur_class
                prev_frame = frame
            elif labels[i, 2] in self.special:
                assert False  # we do not permit codingactive label. though this can be easily supported.
        # extract "exclude" regions
        exclude_regions = self.find_exclude_regions(labels)
        merged_exclude_regions = self.merge_overlapping_intervals(exclude_regions)
        # loop over exclude regions and fix output
        for region in merged_exclude_regions:
            region_start = region[0]
            region_end = region[1]
            # deal with labels before region
            q = [index for index, value in enumerate(output) if value[0] < region_start]
            if q:
                last_overlap = max(q)
                prev_class = output[last_overlap][2]
                output.insert(last_overlap + 1, [region_start, False, prev_class])
            # deal with labels inside region
            q = [index for index, value in enumerate(output) if region_start <= value[0] < region_end]
            if q:
                for index in q:
                    output[index][1] = False
            # deal with last label inside region
            q = [index for index, value in enumerate(output) if value[0] <= region_end]
            if q:
                last_overlap = max(q)
                prev_class = output[last_overlap][2]
                output.insert(last_overlap + 1, [region_end, True, prev_class])
        # finish work
        if not self.return_time_stamps:  # convert to frame numbers
            for entry in output:
                entry[0] = int(int(entry[0]) * self.fps / 1000)
        if not output:  # if nothing was found, all video is invalid
            output.append([0, False, selected_classes[0]])
        start = int(output[0][0])
        trial_times = self.get_trial_intervals(start, labels)
        last_trial_end = trial_times[-1][1]
        annotations_end = int(output[-1][0])
        return output, start, last_trial_end

    def load_and_sort(self, label_path):
        # load label file
        labels = np.genfromtxt(open(label_path, "rb"), dtype='str', delimiter=",", skip_header=3)
        # sort by time
        times = labels[:, 0].astype(np.int)
        sorting_indices = np.argsort(times)
        sorted_labels = labels[sorting_indices]
        return sorted_labels

    def find_exclude_regions(self, sorted_labels):
        regions = []
        for entry in sorted_labels:
            if entry[2] in self.exclude:
                regions.append([int(entry[0]), int(entry[0]) + int(entry[1])])
        return regions

    def get_trial_intervals(self, start, sorted_labels):
        """
        gets trial interval times where beginning is included and end isn't: [)
        :param sorted_labels:
        :return:
        """
        trials = []
        prev_frame = start
        for i in range(len(sorted_labels)):
            if sorted_labels[i, 2] == "end":
                frame_number = int(sorted_labels[i, 0]) + 1  # trial labels are inclusive, i.e. they include last frame.
                if not self.return_time_stamps:  # convert to frame numbers
                    frame = int(frame_number * self.fps / 1000)
                else:
                    frame = frame_number
                trials.append([prev_frame, frame])
                prev_frame = frame
        return trials

    def merge_overlapping_intervals(self, arr):
        merged = []
        if arr:
            arr.sort(key=lambda interval: interval[0])
            merged.append(arr[0])
            for current in arr:
                previous = merged[-1]
                if current[0] <= previous[1]:
                    previous[1] = max(previous[1], current[1])
                else:
                    merged.append(current)
        return merged


class PrefLookTimestampParser(BaseParser):
    """
    a parser that can parse PrefLookTimestamp as described here:
    https://osf.io/3n97m/
    """
    def __init__(self, fps, labels_folder=None, ext=None, return_time_stamps=False):
        super().__init__()
        self.fps = fps
        self.return_time_stamps = return_time_stamps
        if ext:
            self.ext = ext
        if labels_folder:
            self.labels_folder = Path(labels_folder)

    def parse(self, file, file_is_fullpath=False):
        """
        Parses a label file from the lookit dataset, see base class for output format
        :param file: the file to parse
        :param file_is_fullpath: if true, the file represents a full path and extension,
         else uses the initial values provided.
        :return:
        """
        codingactive_counter = 0
        classes = {"away": 0, "left": 1, "right": 2}
        if file_is_fullpath:
            label_path = Path(file)
        else:
            label_path = Path(self.labels_folder, file + self.ext)
        labels = np.genfromtxt(open(label_path, "rb"), dtype='str', delimiter=",", skip_header=3)
        output = []
        start, end = 0, 0
        for entry in range(labels.shape[0]):
            if self.return_time_stamps:
                frame = int(labels[entry, 0])
                dur = int(labels[entry, 1])
            else:
                frame = int(int(labels[entry, 0]) * self.fps / 1000)
                dur = int(int(labels[entry, 1]) * self.fps / 1000)
            class_name = labels[entry, 2]
            valid_flag = 1 if class_name in classes else 0
            if class_name == "codingactive":  # indicates the period of video when coding was actually performed
                codingactive_counter += 1
                start, end = frame, dur
                frame = dur  # if codingactive: add another annotation signaling invalid frames from now on
            frame_label = [frame, valid_flag, class_name]
            output.append(frame_label)
        assert codingactive_counter <= 1  # current parser doesnt support multiple coding active periods
        output.sort(key=lambda x: x[0])
        if end == 0:
            end = int(output[-1][0])
        if len(output) > 0:
            return output, start, end
        else:
            return None


class VCXParser(BaseParser):
    """
    A parser that can parse vcx files that are used in princeton / marchman laboratories
    """
    def __init__(self, fps, raw_dataset_path, raw_dataset_type, first_coder=True):
        super().__init__()
        self.fps = fps
        self.video_dataset = preprocess.build_marchman_video_dataset(raw_dataset_path, raw_dataset_type)
        self.first_coder = first_coder
        self.start_times = self.process_start_times()

    def process_start_times(self):
        start_times = {}
        for entry in self.video_dataset.values():
            if entry["start_timestamp"]:
                time = entry["start_timestamp"]
                time_parts = [int(x) for x in time.split(":")]
                timestamp = time_parts[0]*60*60*self.fps +\
                            time_parts[1]*60*self.fps +\
                            time_parts[2]*self.fps +\
                            time_parts[3]
                start_times[entry["video_id"]] = timestamp
        return start_times

    def parse(self, video_id, label_path=None):
        """
        parse a coding file, see base class for output format
        :param video_id: video_id of video
        :param label_path: if provided, will parse this file instead
        :return: see base class
        """
        if not label_path:
            if self.first_coder:
                label_path = self.video_dataset[video_id]["first_coding_file"]
            else:
                label_path = self.video_dataset[video_id]["second_coding_file"]
            if not label_path.is_file():
                logging.warning("For the file: " + str(label_path) + " no matching vcx was found.")
                return None
        return self.xml_parse(video_id, label_path)

    def xml_parse(self, video_id, input_file):
        tree = ET.parse(input_file)
        root = tree.getroot()
        # find "Responses" child, and return the child right after it.
        flag = False
        for child in root.iter('*'):
            if flag:
                responses_element = child
                break
            if child.text is not None:
                if "Responses" == child.text:
                    flag = True
        # iterate over children, creating a response string
        state = 0
        responses = []
        for child in responses_element:
            if state == 0:  # new response
                response_list = []
                response_list.append(child.text)
                state = 1
            elif state == 1:  # inside a response
                for gchild in child.iter("*"):
                    if gchild.text is not None:
                        response_list.append(gchild.text)
                    else:
                        response_list.append(gchild.tag)
                responses.append(self.parse_response_list(response_list))
                state = 0
        # sort by response index
        sorted_responses = sorted(responses)
        # assemble final response list as required by parser API
        final_responses = []
        cur_timestamp = -1
        for response in sorted_responses:
            timestamp = response[1]
            assert timestamp > cur_timestamp, "Can't Have two responses for the same timestamp !"
            cur_timestamp = timestamp
            status = response[2]
            label = response[3]
            if label == 'off' or label == 'center':
                label = 'away'
            if self.start_times:
                start_time = self.start_times[video_id]
                timestamp -= start_time
            assert 0 <= timestamp < 60 * 60 * self.fps, "Starting time provided is after first response !"
            final_responses.append([timestamp, status, label])
        assert len(final_responses) != 0, "No responses in file !"
        start = final_responses[0][0]
        intervals = self.get_trial_intervals(start, final_responses)
        end = intervals[-1][1]
        return final_responses, start, end

    def parse_response_list(self, response_array):
        response_index = response_array[0]
        response_index = int(response_index.split()[-1])
        frame = response_array[response_array.index("Frame") + 1]
        hour = response_array[response_array.index("Hour") + 1]
        minute = response_array[response_array.index("Minute") + 1]
        second = response_array[response_array.index("Second") + 1]
        try:
            status = response_array[response_array.index("Trial Status") + 1]
        except ValueError:
            status = response_array[response_array.index("Slide") + 1]
        label = response_array[response_array.index("Type") + 1]
        timestamp = int(frame) +\
                    int(second) * self.fps +\
                    int(minute) * 60 * self.fps +\
                    int(hour) * 60 * 60 * self.fps
        return [response_index, timestamp, int(status.lower() == "true"), label]

    def get_trial_intervals(self, start, responses):
        """
        gets trial ending times, in a non-inclusive manner
         i.e. open ended interval [)
        :param label_path: path to label file
        :return:
        """
        trials_times = []
        prev_frame = start
        for response in responses:
            if response[1] == 0:
                trials_times.append([prev_frame, response[0]])
                prev_frame = response[0]
        return trials_times


class DatavyuParser(BaseParser):
    """
    parses datavyu files
    """
    def __init__(self):
        super().__init__()

    def parse(self, video_id, label_path):
        if label_path:  # AW/LB are the coder initials, and they used different column names unfortunately
            data = pd.read_csv(label_path)
            coding = np.ones([len(data), 1])  # on looks are not coded (implied by other conditions)
            coding[data['look_type'] == 'n'] = 0  # off looks
            coding[data['look_type'] == 'e'] = -1  # error looks
            coding = coding.squeeze()
            trials = self.get_trial_intervals(0, label_path)
            coding[:trials[0][0]] = -1  # mark all coding until first trial as invalid
            return coding, trials[0][0], trials[-1][1]
        else:
            return None

    def get_trial_intervals(self, start, label_path):
        if label_path:
            data = pd.read_csv(label_path)  # read data
            data = data[data['trial_type'] != 'a']  # remove attention getter trials (haven't been coded meticulously)
            onset_col = data['trial_onset']
            onset_times = np.unique(onset_col)
            onset_frames = [data['nFrame'].iloc[np.where(onset_col==time)[0][0]] for time in onset_times[~np.isnan(onset_times)]]  # first frame of trial
            offset_col = data['trial_offset']  # get offset column
            offset_times = np.unique(offset_col)  # unique offset times, add one
            offset_frames = [1 + data['nFrame'].iloc[np.where(offset_col==time)[0][-1]] for time in offset_times[~np.isnan(offset_times)]]  # last frame of trial
            zipped_trial_times = list(np.dstack([onset_frames, offset_frames]).flatten())  # interleave onsets and offsets
            zipped_trial_times = [zipped_trial_times[i: i+2] for i in range(0, len(zipped_trial_times), 2)]  # convert to list of lists
            return zipped_trial_times
        else:
            return None

def parse_illegal_transitions_file(path, skip_header=True):
    illegal_transitions = []
    corrected_transitions = []
    if path is not None:
        with open(path, newline='') as f:
            rows = f.readlines()
        # skip header
        if skip_header:
            rows = rows[1:]
        for row in rows:
            row = row.split(",")
            row = [x.strip() for x in row]
            if len(row) != 2:
                raise ValueError("Illegal transitions file needs to have exactly two columns.")
            ilegal, corrected = row
            try:
                illegal_transitions.append([int(x) for x in ilegal])
                corrected_transitions.append([int(x) for x in corrected])
            except ValueError:
                raise ValueError("Illegal transitions file needs to have only integers.")
    return illegal_transitions, corrected_transitions