[![DOI](https://zenodo.org/badge/486841882.svg)](https://zenodo.org/badge/latestdoi/486841882)
# Introduction
This repository contains all the code for [iCatcher+](https://psyarxiv.com/up97k/), a tool for performing automatic annotation of discrete infant gaze directions from videos collected in the lab, field or online (remotely). It also contains code for reproducing the original manuscripts results.

Click below for a video including examples of representative good and poor performance, taken from videos of infants participating in online research (all families featured consented to sharing their video data publicly):

[![iCatcher representative good and poor performance](https://img.youtube.com/vi/iK_T2P2ZDnU/0.jpg)](https://www.youtube.com/watch?v=iK_T2P2ZDnU)


# Installation

You can set up your own conda environment, or access the environment/code/models wrapped up as a [container](https://apptainer.org/user-docs/3.8/introduction.html#why-use-containers). Instructions for both options are below. 

## Setup with conda environment

### Step 1: Clone this repository to get a copy of the code to run locally

`git clone https://github.com/yoterel/icatcher_plus.git`

### Step 2: Create a conda virtual environment

We recommend installing [Miniconda](https://docs.conda.io/en/latest/miniconda.html) for this, but you can also [Install Anaconda](https://www.anaconda.com/products/individual/get-started) if needed, then create an environment using the environment.yml file in this repository:

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

## Setup with Docker container 

A Docker image is now available here: https://hub.docker.com/repository/docker/saxelab/looking_time_analysis 

### Use as [Docker](https://aws.amazon.com/docker/) container (e.g., to run on your own computer):

#### Step 1: Install Docker

Follow instructions on the [Docker site](https://docs.docker.com/get-docker/) to install Docker Desktop on your computer 

#### Step 2: Pull a copy of the docker container for use

`docker pull saxelab/looking_time_analysis:icatcher_env`

### Use as [Singularity](https://apptainer.org/user-docs/3.8/) container: 

Singularity is the main containerization software that is used on High-Performance Computing Clusters (HPCCs), but is a similar containerization software to Docker. If you use Singularity instead of docker, you can convert the Docker image to a singularity container using the following command: 

#### Step 2: pull the Docker image into a singularity container:

##### Directly to an immutable container: 

`singularity build icatcher_env.sif docker://saxelab/looking_time_analysis:icatcher_env`

##### .. Or as a "sandbox" for editing: 

If you want to make changes to the environment, you can also build as a sandbox: 

`singularity build --sandbox icatcher_env/ docker://saxelab/looking_time_analysis:icatcher_env`

then use `singularity shell --writable icatcher_env` to access the sandbox container for making changes

and finally, `singularity build icatcher_env.sif icatcher_env/` to make the container immutable. 


# Running iCatcher+

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

### Running the code from a container, e.g., as a singularity container: 

`singularity exec -B /filesystem/filesystem icatcher_env.sif python3 /icatcher_plus/test.py /path/to/my/video.mp4  /icatcher_plus/models/icatcher+_lookit.pth --fc_model /icatcher_plus/models/face_classifier_lookit.pth --output_annotation /path/to/output_folder --output_format compressed --output_video_path /path/to/output_folder`


# Output format

The test.py file currently supports 3 output formats, though further formats can be added upon request.

- raw_output: a file where each row will contain the frame number, the class prediction and the confidence of that prediction seperated by a comma
- compressed: a npz file containing two numpy arrays, one encoding the predicted class (n x 1 int32) and another the confidence (n x 1 float32) where n is the number of frames. This file can be loaded into memory using the numpy.load function. For the map between class number and name see test.py ("predict_from_video" function).
- PrefLookTimestamp: will save a file in the format described [here](https://osf.io/3n97m/) describing the output of the automated coding.

# Datasets access & reproduction of results

The public videos from the Lookit dataset, along with human annotations and group-level demographics for all datasets, are available at https://osf.io/ujteb/. Videos from the Lookit dataset with permission granted for scientific use are available at https://osf.io/5u9df/. Requests for access can be directed to Junyi Chu (junyichu@mit.edu).

Requests for access to the remainder of the datasets (Cali-BW, Senegal) can be directed to Dr. Katherine Adams Shannon (katashannon@gmail.com). Note that access to raw video files from the California-BW and Senegal datasets *is not available* due to restricted participant privacy agreements. To protect participant privacy, the participant identifiers for the video and demographic data are not linked to each other. However, this information is available upon reasonable request.

We made substantial effort to allow reproduction of results form the paper. True reproduction requires full access to the datasets (including the videos).
Instead, to reproduce most of the statistics we present in the paper for the Lookit dataset, run visualize.py using the following commands:

First navigate to where you placed the source code at:

`cd /path/to/icathcer_plus`

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


