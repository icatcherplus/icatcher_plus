[![DOI](https://zenodo.org/badge/486841882.svg)](https://zenodo.org/badge/latestdoi/486841882)

# iCatcher+

# Introduction
This repository contains the official code for [iCatcher+](https://doi.org/10.1177/25152459221147250), a tool for performing automatic annotation of discrete infant gaze directions from videos collected in the lab, field or online (remotely). It also contains code for reproducing the original paper results.

Click below for a video including examples of representative good and poor performance, taken from videos of infants participating in online research (all families featured consented to sharing their video data publicly):

[![iCatcher representative good and poor performance](https://img.youtube.com/vi/iK_T2P2ZDnU/0.jpg)](https://www.youtube.com/watch?v=iK_T2P2ZDnU)

# Installation
## Quick installation (Windows, Linux, Mac)
This option will let you use iCatcher+ with minimum effort, but only for predictions (inference).
We strongly recommend using a virtual environment such as [Miniconda](https://conda.io) or [virtualenv](https://pypi.org/project/virtualenv/) before running the command below.

`pip install icatcher`

You will also need [ffmpeg](https://www.ffmpeg.org/) installed in your system and available (if you used conda, you can quickly install it with `conda install -c conda-forge ffmpeg`).

Note1:
If you require speedy performance, prior to installing icatcher you should install [PyTorch](https://pytorch.org/) with GPU support (see [here](https://pytorch.org/get-started/locally/) for instructions). This assumes you have a supported GPU on your machine.

Note2:
When using iCatcher+ for the first time, neural network model files will automatically be downloaded to a local cache folder. To control where they are downloaded to set the "ICATCHER_DATA_DIR" environment variable.

## Reproduction of original research results / retraining on your own dataset

see [reproduce](https://github.com/icatcherplus/icatcher_plus/tree/master/reproduce) for a full set of instructions.

# Running iCatcher+

You can run iCatcher+ with the command:

`icatcher --help`

which will list all available options. Description below will help you get more familiar with some common command line arguments.

To run iCatcher+ with a video file (if a folder is provided, all videos will be used for prediction):

`icatcher /path/to/my/video.mp4`

A common option is to add:

`icatcher /path/to/my/video.mp4 --use_fc_model`

Which enables a child face detector for more robust results (however, sometimes this can result in too much loss of data).

You can save a labeled video by adding:

`--output_video_path /path/to/output_folder`

If you want to output annotations to a file, use:

`--output_annotation /path/to/output_annotation_folder`

To show the predictions online in a seperate window, add the option:

`--show_output`

You can also add parameters to crop the video a given percent before passing to iCatcher: 

`--crop_mode m` where `m` is any of [top, left, right], specifying which side of the video to crop from (if not provided, default is none; if crop_percent is provided but not crop_mode, default is top)

`--crop_percent x` where `x` is an integer (default = 0) specifying what percent of the video size to crop from the specified side. E.g., if `--crop_mode top` is provided with `--crop_percent 10`, 10% of the video height will be cropped from the top. If `--crop_mode left` is provided with `--crop_percent 25`, 25% of the video width will be cropped from the left side, etc. 

# Output format

Currently we supports 2 output formats, though further formats can be added upon request.

- raw_output: a file where each row will contain the frame number, the class prediction and the confidence of that prediction seperated by a comma
- compressed: a npz file containing two numpy arrays, one encoding the predicted class (n x 1 int32) and another the confidence (n x 1 float32) where n is the number of frames. This file can be loaded into memory using the numpy.load function. For the map between class number and name see test.py ("predict_from_video" function).

# Datasets access

The public videos from the Lookit dataset, along with human annotations and group-level demographics for all datasets, are available at https://osf.io/ujteb/. Videos from the Lookit dataset with permission granted for scientific use are available at https://osf.io/5u9df/. Requests for access can be directed to Junyi Chu (junyichu@mit.edu).

Requests for access to the remainder of the datasets (Cali-BW, Senegal) can be directed to Dr. Katherine Adams Shannon (katashannon@gmail.com). Note that access to raw video files from the California-BW and Senegal datasets *is not available* due to restricted participant privacy agreements. To protect participant privacy, the participant identifiers for the video and demographic data are not linked to each other. However, this information is available upon reasonable request.

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

    ├── src                     # code for package (inference only)
    ├── tests                   # tests for package
    ├── reproduce               # all code used for producing paper results, including training and visualizations.
    
# Troubleshooting Issues
Please open a github issue for any question or problem you encounter. We kindly ask to first skim through closed issues to see if your problem was already addressed.

# Citation
```
@article{doi:10.1177/25152459221147250,
author = {Yotam Erel and Katherine Adams Shannon and Junyi Chu and Kim Scott and Melissa Kline Struhl and Peng Cao and Xincheng Tan and Peter Hart and Gal Raz and Sabrina Piccolo and Catherine Mei and Christine Potter and Sagi Jaffe-Dax and Casey Lew-Williams and Joshua Tenenbaum and Katherine Fairchild and Amit Bermano and Shari Liu},
title ={iCatcher+: Robust and Automated Annotation of Infants’ and Young Children’s Gaze Behavior From Videos Collected in Laboratory, Field, and Online Studies},
journal = {Advances in Methods and Practices in Psychological Science},
volume = {6},
number = {2},
pages = {25152459221147250},
year = {2023},
doi = {10.1177/25152459221147250},
URL = { 
        https://doi.org/10.1177/25152459221147250
},
eprint = { 
        https://doi.org/10.1177/25152459221147250 
}
}
```


