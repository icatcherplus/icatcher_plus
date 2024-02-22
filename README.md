[![Generic badge](https://img.shields.io/badge/Website-Online-Green.svg)](https://icatcherplus.github.io/) [![PyPI version](https://badge.fury.io/py/icatcher.svg)](https://badge.fury.io/py/icatcher) [![Test iCatcher+](https://github.com/icatcherplus/icatcher_plus/actions/workflows/test.yml/badge.svg)](https://github.com/icatcherplus/icatcher_plus/actions/workflows/test.yml) [![DOI](https://zenodo.org/badge/486841882.svg)](https://zenodo.org/badge/latestdoi/486841882)

# iCatcher+

# Introduction
This repository contains the official code for [iCatcher+](https://doi.org/10.1177/25152459221147250), a tool for performing automatic annotation of discrete infant gaze directions from videos collected in the lab, field or online (remotely). 

The codebase comprises three parts: 
1. A Python-based ML tool for generating gaze annotations
2. A browser-based web app for reviewing the generated annotations
3. Code for reproducing the original paper results

Click below for a video including examples of representative good and poor performance, taken from videos of infants participating in online research (all families featured consented to sharing their video data publicly):

[![iCatcher representative good and poor performance](https://img.youtube.com/vi/iK_T2P2ZDnU/0.jpg)](https://www.youtube.com/watch?v=iK_T2P2ZDnU)

# Installation
## Quick installation (Windows, Linux, Mac)
This option will install the most up-to-date version of the iCatcher+ annotation tool and web app with minimum effort. However, it will not provide the code to reproduce the original paper results or train your own model. For instructions on how to reproduce, see [here](https://github.com/icatcherplus/icatcher_plus/tree/master/reproduce).

`pip install icatcher`

We strongly recommend using a virtual environment such as [Miniconda](https://conda.io) or [virtualenv](https://pypi.org/project/virtualenv/) before running the command above.

You will also need [ffmpeg](https://www.ffmpeg.org/) installed in your system and available (if you used conda, you can quickly install it with `conda install -c conda-forge ffmpeg`).

Note 1:

If you require speedy performance, prior to installing icatcher you should install [PyTorch](https://pytorch.org/) with GPU support (see [here](https://pytorch.org/get-started/locally/) for instructions). This assumes you have a supported GPU on your machine.

Note 2:

When using iCatcher+ for the first time, neural network model files will automatically be downloaded to a local cache folder. To control where they are downloaded to set the "ICATCHER_DATA_DIR" environment variable.

## Reproduction of original research results / retraining on your own dataset

see [reproduce](https://github.com/icatcherplus/icatcher_plus/tree/master/reproduce) for a full set of instructions.

## Developer Install
If installed via `git clone`, extra steps need to be taken to set up the web app. See [src/icatcher/icatcher_app](src/icatcher/icatcher_app/README.md) for full instructions.

# Running iCatcher+

You can run iCatcher+ with the command:

`icatcher --help`

which will list all available options. The description below will help you get more familiar with some common command line arguments.

### Annotating a Video
To produce annotations for a video file (if a folder is provided, all videos will be used for prediction):

`icatcher /path/to/my/video.mp4`

**NOTE:** For any videos you wish to visualize with the [Web App](#web-app), you must use the `--output_annotation` and the `--output_format ui` flags:

`icatcher /path/to/my/video.mp4 --output_annotation /path/to/desired/output/directory/ --output_format ui`

### Common Flags

You can save a labeled video by adding:

`--output_video_path /path/to/output_folder`

If you want to output annotations to a file, use:

`--output_annotation /path/to/output_annotation_folder`

See [Output format](#output-format) below for more information on how the files are formatted.

To show the predictions online in a seperate window, add the option:

`--show_output`

To launch the iCatcher+ web app (after annotating), use:

`icatcher --app`

The app should open automatically at [http://localhost:5001](http://localhost:5001). For more details, see [Web App](#web-app).

Originally a face classifier was used to distinguish between adult and infant faces (however this can result in too much loss of data). It can be turned on by using:

`icatcher /path/to/my/video.mp4 --use_fc_model`

You can also add parameters to crop the video a given percent before passing to iCatcher: 

`--crop_mode m` where `m` is any of [top, left, right], specifying which side of the video to crop from (if not provided, default is none; if crop_percent is provided but not crop_mode, default is top)

`--crop_percent x` where `x` is an integer (default = 0) specifying what percent of the video size to crop from the specified side. E.g., if `--crop_mode top` is provided with `--crop_percent 10`, 10% of the video height will be cropped from the top. If `--crop_mode left` is provided with `--crop_percent 25`, 25% of the video width will be cropped from the left side, etc. 

# Output format

Currently we supports 3 output formats, though further formats can be added upon request.

- **raw_output:** a file where each row will contain the frame number, the class prediction and the confidence of that prediction seperated by a comma
- **compressed:** a npz file containing two numpy arrays, one encoding the predicted class (n x 1 int32) and another the confidence (n x 1 float32) where n is the number of frames. This file can be loaded into memory using the numpy.load function. For the map between class number and name see test.py ("predict_from_video" function).
- **ui:** needed to open a video in the web app; produces a directory of the following structure

        ├── decorated_frames     # dir containing annotated jpg files for each frame in the video
        ├── video.mp4            # the original video
        ├── labels.txt           # file containing annotations in the `raw_output` format described above

# Web App
The iCatcher+ app is a tool that allows users to interact with output from the iCatcher+ ML pipeline in the browser. The tool is designed to operate entirely locally and will not upload any input files to remote servers.

### Using the UI

When you open the iCatcher+ UI, you will be met with a pop-up inviting you to upload a directory. Please note, this requires you to upload *the whole output directory* which should include a `labels.txt` file and a sub-directory named `decorated_frames` containing all of the frames of the video as image files.

Once you've uploaded a directory, you should see a pop-up asking whether you are sure want to upload all files. Rest assured, this will not upload the files to any remote servers. This is only giving the local browser permission to access those files. The files will stay local to whatever computer is running the browser.

At this point, you should see the video on the screen (you may need to give it a few second to load). Now you can start to review the annotations. Below the video you'll see heatmaps giving you a visual overview of the labels for each frame, as well as the confidence level for each frame.

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
        ├── icatcher_app        # code for web app
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


