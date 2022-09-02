## Introduction
This repository contains all the code for [iCatcher+](https://psyarxiv.com/up97k/), a tool for performing automatic annotation of discrete infant gaze directions from videos collected in the lab or online (remotely). It also contains code for reproducing the original manuscripts results.

## Installation

### Step 1: Clone this repository to get a copy of the code to run locally

`git clone https://github.com/yoterel/icatcher_plus.git`

### Step 2: Create a conda virtual environment

We recommend installing [Miniconda](https://docs.conda.io/en/latest/miniconda.html) for this, but you can also [Install Anaconda](https://www.anaconda.com/products/individual/get-started) if needed, then create an environment using the environment.yml file in this repository:

**Note**: conda must be in the PATH envrionment variable for the shell to find it.

`conda env create -n env -f environment.yml`

or

`conda env create --prefix /path/to/virtual/environment -f "/path/to/environment.yml"`

Activate the environment

`conda activate env`

Navigate to the icatcher_plus folder using a terminal / command prompt:

`cd icatcher_plus`

### Step 3: Download the latest network model & weights file

iCatcher+ relies on some neural-network model files to work (or reproduce experiments).

Please download all files from [here](https://www.cs.tau.ac.il/~yotamerel/icatcher+/icatcher+_models.zip) and place them in the models directory.


### Step 4: Running iCatcher+

To run icatcher with a video file (if a folder is provided, all videos will be used for prediction):

`python test.py /path/to/my/video.mp4 /path/to/icatcher_model.pth --fc_model /path/to/face_classifier.pth`

You can save a labeled video by adding:

`--output_video_path /path/to/output_folder`

If you want to output annotations to a file, use:

`--output_annotation /path/to/output_annotation_folder`

To show the predictions online in a seperate window, add the option:

`--show_output`

For a full command line option list use:

`python test.py --help`

## Output format

The test.py file currently supports 3 output formats, though further formats can be added upon request.

- raw_output: a file where each row will contain the frame number, the class prediction and the confidence of that prediction seperated by a comma
- compressed: a npz file containing two numpy arrays, one encoding the predicted class (n x 1 int32) and another the confidence (n x 1 float32) where n is the number of frames. This file can be loaded into memory using the numpy.load function. For the map between class number and name see test.py ("predict_from_video" function).
- PrefLookTimestamp: will save a file in the format described [here](https://osf.io/3n97m/) describing the output of the automated coding.

## Datasets access

The public videos from the Lookit dataset, along with human annotations and group-level demographics for all datasets, are available at https://osf.io/ujteb/. Videos from the Lookit dataset with permission granted for scientific use are available at https://osf.io/5u9df/. Requests for access can be directed to Junyi Chu (junyichu@mit.edu).

Requests for access to the remainder of the datasets (Cali-BW, Senegal) can be directed to Dr. Katherine Adams Shannon (katashannon@gmail.com). Note that access to raw video files from the California-BW and Senegal datasets *is not available* due to restricted participant privacy agreements. To protect participant privacy, the participant identifiers for the video and demographic data are not linked to each other. However, this information is also available upon reasonable request.


## Best Results (test sets)
To view all results, see all plots under /plots.
per session plots (per-video) are sorted from 0 to n, where 0 has the lowest agreement (between iCatcher+ and Coder 1) and n the highest.

<table>
        <tr>
                <td align="center"> <img src="https://github.com/yoterel/icatcher_plus/blob/master/assets/agreement.png"  alt="0" width = 400px height = 300px ></td>
                <td align="center"><img src="https://github.com/yoterel/icatcher_plus/blob/master/assets/agreement_vs_confidence.png"  alt="0" width = 400px height = 300px ></td>
        </tr>
        <tr><td colspan=2>California-BW</td></tr>
        <tr>
                <td><img src="https://github.com/yoterel/icatcher_plus/blob/master/assets/cali-bw_bar.png" alt="0" width = 400px height = 300px></td>
                <td><img src="https://github.com/yoterel/icatcher_plus/blob/master/assets/cali-bw_conf.png" alt="1" width = 300px height = 300px></td>
        </tr>
        <tr><td colspan=2>Lookit</td></tr>
        <tr>
                <td><img src="https://github.com/yoterel/icatcher_plus/blob/master/assets/lookit_bar.png" alt="0" width = 400px height = 300px></td>
                <td><img src="https://github.com/yoterel/icatcher_plus/blob/master/assets/lookit_conf.png" alt="1" width = 300px height = 300px></td>
        </tr>
</table>

## Project Structure (subject to change):


    ├── assets                  # contains assets for README
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
    
## Contributions
Feel free to contribute by submitting a pull request. Make sure to run all tests under /tests


