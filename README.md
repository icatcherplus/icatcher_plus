[![DOI](https://zenodo.org/badge/486841882.svg)](https://zenodo.org/badge/latestdoi/486841882)

# iCatcher+ v0.0.4

# Introduction
This repository contains all the code for [iCatcher+](https://psyarxiv.com/up97k/), a tool for performing automatic annotation of discrete infant gaze directions from videos collected in the lab, field or online (remotely). It also contains code for reproducing the original manuscripts results.

Click below for a video including examples of representative good and poor performance, taken from videos of infants participating in online research (all families featured consented to sharing their video data publicly):

[![iCatcher representative good and poor performance](https://img.youtube.com/vi/iK_T2P2ZDnU/0.jpg)](https://www.youtube.com/watch?v=iK_T2P2ZDnU)

# Installation
## Quick installation (Windows & Linux only)
This option will let you use iCatcher+ with minimum effort, but only for predictions (inference).
We strongly recommend using a virtual environment such as [Miniconda](https://conda.io) or [virtualenv](https://pypi.org/project/virtualenv/) before running the command below.

`pip install icatcher`

You will also need [ffmpeg](https://www.ffmpeg.org/) installed in your system and available, as well as model files:
Please download all files from [here](https://www.cs.tau.ac.il/~yotamerel/icatcher+/icatcher+_models.zip) and place them in a directory named models. iCatcher+ expects this folder to exist in the location you launch it from (see Running iCatcher+)

## Installation from source
This options allows you to reproduce the paper results, train icatcher on your own dataset, and tinker with its core.

### Step 1: Clone this repository to get a copy of the code to run locally

`git clone https://github.com/yoterel/icatcher_plus.git`

### Step 2: Create a conda virtual environment

We recommend installing [Miniconda](https://docs.conda.io/en/latest/miniconda.html) for this, then create an environment using the environment.yml file in this repository:

**Note**: conda must be in the PATH envrionment variable for the shell to find it.

Navigate to the icatcher_plus folder using the Anaconda Prompt:

`cd /path/to/icathcer_plus`

Then:

`conda env create -n env -f environment.yml`

Or (if you want to install the environment in a specific location):

`conda env create --prefix /path/to/virtual/environment -f "/path/to/environment.yml"`

**Note for Mac users**: you might need to edit the [environment.yml](https://github.com/yoterel/icatcher_plus/blob/master/environment.yml) file depending on your OS version. see [here](https://github.com/yoterel/icatcher_plus/issues/6#issuecomment-1244125700) for how.

Activate the environment

`conda activate env`

### Step 3: Download the latest network model & weights file

iCatcher+ relies on some neural-network model files to work (or reproduce experiments).

Please download all files from [here](https://www.cs.tau.ac.il/~yotamerel/icatcher+/icatcher+_models.zip) and place them in the models directory.


# Running iCatcher+
If you installed iCatcher+ using the quick installation, you can run it with the command:
`icatcher --help`
which will list all available options. Description below will help you with some common command line arguments, but you need to replace `python test.py` with `icatcher` for it to work.

To run icatcher with a video file (if a folder is provided, all videos will be used for prediction):

`python test.py /path/to/my/video.mp4 /path/to/icatcher_model.pth --fc_model /path/to/face_classifier.pth`

You can save a labeled video by adding:

`--output_video_path /path/to/output_folder`

If you want to output annotations to a file, use:

`--output_annotation /path/to/output_annotation_folder`

To show the predictions online in a seperate window, add the option:

`--show_output`

You can also add parameters to crop the video before passing to iCatcher: 

`--crop_mode m` where `m` is any of [top, left, right], specifying which side of the video to crop from (if not provided, default is none; if crop_percent is provided but not crop_mode, default is top)

For a full command line option list (there are several other optional parameters!) use:

`python test.py --help`

# Output format

Currently we supports 3 output formats, though further formats can be added upon request.

- raw_output: a file where each row will contain the frame number, the class prediction and the confidence of that prediction seperated by a comma
- compressed: a npz file containing two numpy arrays, one encoding the predicted class (n x 1 int32) and another the confidence (n x 1 float32) where n is the number of frames. This file can be loaded into memory using the numpy.load function. For the map between class number and name see test.py ("predict_from_video" function).
- PrefLookTimestamp: will save a file in the format described [here](https://osf.io/3n97m/) describing the output of the automated coding.

# Datasets access & reproduction of results

The public videos from the Lookit dataset, along with human annotations and group-level demographics for all datasets, are available at https://osf.io/ujteb/. Videos from the Lookit dataset with permission granted for scientific use are available at https://osf.io/5u9df/. Requests for access can be directed to Junyi Chu (junyichu@mit.edu).

Requests for access to the remainder of the datasets (Cali-BW, Senegal) can be directed to Dr. Katherine Adams Shannon (katashannon@gmail.com). Note that access to raw video files from the California-BW and Senegal datasets *is not available* due to restricted participant privacy agreements. To protect participant privacy, the participant identifiers for the video and demographic data are not linked to each other. However, this information is available upon reasonable request.

We made substantial effort to allow reproduction of results form the paper. True reproduction requires full access to the datasets (including the videos).
Instead, to reproduce most of the statistics we present in the paper for the Lookit dataset, run visualize.py using the following commands:

First navigate to where you placed the source code at:

`cd /path/to/icatcher_plus`

Then run:

`python visualize.py output resource/lookit_annotations/coding_human1 resource/lookit_annotations/coding_icatcherplus just_annotations --human2_codings_folder resource/lookit_annotations/coding_human2`

Results will appear in a folder called "output".

## Best Results (test sets)
To view visualizations of all results, see [plots](https://github.com/yoterel/icatcher_plus/tree/master/plots).
Per-session plots (i.e. per-video) are sorted from 0 to n, where 0 has the lowest agreement (between iCatcher+ and Coder 1) and n the highest.
### A Note About Data-Leaks
The test sets were kept "untouched" until the very last stages of submission (i.e. they were not *directly* nor *indirectly* used optimize the network models). Conforming to this methodolgy is encouraged to avoid data leaks, so if you happen to submit improvements made to iCatcher+ in terms of performance, **do not** use the test sets for improving your method. Please consider creating a validation set out of the original training set for that.

<table>
        <tr>
                <td align="center"> <img src="https://github.com/yoterel/icatcher_plus/blob/master/resource/agreement.png"  alt="0" width = 400px height = 300px ></td>
                <td align="center"><img src="https://github.com/yoterel/icatcher_plus/blob/master/resource/agreement_vs_confidence.png"  alt="0" width = 400px height = 300px ></td>
        </tr>
        <tr><td colspan=2>Lookit</td></tr>
        <tr>
                <td><img src="https://github.com/yoterel/icatcher_plus/blob/master/resource/lookit_bar.png" alt="0" width = 400px height = 300px></td>
                <td><img src="https://github.com/yoterel/icatcher_plus/blob/master/resource/lookit_conf.png" alt="1" width = 300px height = 300px></td>
        </tr>
        <tr><td colspan=2>California-BW</td></tr>
        <tr>
                <td><img src="https://github.com/yoterel/icatcher_plus/blob/master/resource/cali-bw_bar.png" alt="0" width = 400px height = 300px></td>
                <td><img src="https://github.com/yoterel/icatcher_plus/blob/master/resource/cali-bw_conf.png" alt="1" width = 300px height = 300px></td>
        </tr>
        <tr><td colspan=2>Senegal</td></tr>
        <tr>
                <td><img src="https://github.com/yoterel/icatcher_plus/blob/master/resource/senegal_bar.png" alt="0" width = 400px height = 300px></td>
                <td><img src="https://github.com/yoterel/icatcher_plus/blob/master/resource/senegal_conf.png" alt="1" width = 300px height = 300px></td>
        </tr>
</table>


# Performance Benchmark
We benchmarked iCatcher+ performance over 10 videos (res 640 x 480). Reported results are averaged upon all frames.

<table>
        <tr>
                <td>iCatcher+ on GPU (NVIDIA GeForce RTX 2060)</td>
                <td>~45 fps</td>
        </tr>
        <tr>
                <td>iCatcher+ on CPU (Intel Core i7-9700)</td>
                <td>~17 fps</td>
        </tr>
</table>

## Project Structure (subject to change):


    ├── resource                # contains extra resources
    ├── datasets                # place holder for datasets 
    ├── face_classifier         # contains all specific code for face classification, separated from main project files on purpose.
        ├── fc_data.py          # creates face classifier dataset
        ├── fc_eval.py          # face classifier evaluation
        ├── fc_model.py         # face classifier model
        ├── fc_train.py         # face classifier training  script
    ├── models                  # place holder for model files
    ├── plots                   # place holder for various plots
    ├── statistics              # code for analyzing multi-variant video dataset statistics
    ├── tests                   # pytests
    ├── augmentations.py        # defines RandAugment set of augmentations
    ├── data.py                 # dataloaders and datasets
    ├── logger.py               # simple logger class
    ├── models.py               # definition of model architectures etc
    ├── options.py              # parse command line arguments
    ├── parsers.py              # annotations (labels) of videos in various formats are parsed using classes in this file
    ├── preprocess.py           # used to parse a raw dataset from OSF into a dataloader ready dataset
    ├── test.py                 # use this to run a full test on a video or a folder of videos
    ├── train.py                # main training loop
    ├── video.py                # API to ffmpeg functionallity
    ├── visualize.py            # compares human annotation with results from icatcher to reproduce paper results
    
# Contributions
Feel free to contribute by submitting a pull request. Make sure to run all tests under /tests


