This folder allows you to reproduce the paper results, train icatcher on your own dataset, compare and measure performance and visualize results as well.

# Getting a local copy of the code
This section is mandatory before any reproduction / retraining /customization can occur.

### Step 1: Clone the repository to get a copy of the code to run locally

`git clone https://github.com/yoterel/icatcher_plus.git`

### Step 2: Create a conda virtual environment

We recommend installing [Miniconda](https://docs.conda.io/en/latest/miniconda.html) for this, then create an environment using the environment.yml file in this repository:

**Note**: conda must be in the PATH envrionment variable for the shell to find it.

Navigate to the icatcher_plus reproduce folder using the Anaconda Prompt:

`cd /path/to/icathcer_plus/reproduce`

Then:

`conda env create -n env -f environment.yml`

Or (if you want to install the environment in a specific location):

`conda env create --prefix /path/to/virtual/environment -f "/path/to/environment.yml"`

**Note for Mac users**: you might need to edit the [environment.yml](https://github.com/yoterel/icatcher_plus/blob/master/environment.yml) file depending on your OS version. see [here](https://github.com/yoterel/icatcher_plus/issues/6#issuecomment-1244125700) for how.

Activate the environment

`conda activate env`

### Step 3: Download the neural network model & weight files

iCatcher+ relies on some neural-network model files to work (or reproduce experiments).

Please download all files from [here](https://osf.io/ycju8/download) and place them in the reproduce/models directory.

# Reproduction 
This section explains how to reproduce original manuscript results.
We made substantial effort to allow reproduction of results form the paper. However, true reproduction requires full access to the datasets (including the videos).
Instead, to reproduce most of the statistics we present in the paper for the Lookit dataset, run visualize.py using the following commands:

First navigate to the reproduce folder:

`cd /path/to/icatcher_plus/reproduce`

Then run:

`python visualize.py output resource/lookit_annotations/coding_human1 resource/lookit_annotations/coding_icatcherplus just_annotations --human2_codings_folder resource/lookit_annotations/coding_human2`

Results will appear in a folder called "output".

### Best Results (test sets)
To view visualizations of all results, see [plots](https://github.com/yoterel/icatcher_plus/tree/master/reproduce/plots).
Per-session plots (i.e. per-video) are sorted from 0 to n, where 0 has the lowest agreement (between iCatcher+ and Coder 1) and n the highest.
- A Note About Data-Leaks: the test sets were kept "untouched" until the very last stages of submission (i.e. they were not *directly* nor *indirectly* used optimize the network models). Conforming to this methodolgy is encouraged to avoid data leaks, so if you happen to submit improvements made to iCatcher+ in terms of performance, **do not** use the test sets for improving your method. Please consider creating a validation set out of the original training set for that.

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

# Training on your own dataset
This section will explain how to train iCatcher+ on your own dataset, or the datasets we used in the manuscript, but please note this requires some dive ins into the code, and is highly experimental.

## Preprocessing
Before any training can happen, a mandatory preprocessing step must occur.
The goal of this step is to process the raw dataset into data that the training dataloader can make sense of, and is usually the most "troublesome" part of the training procedure, as every raw dataset is slightly different.

Run preprocess.py on the lookit dataset and observe how the raw dataset is processed into the correct format. The point you will probably have to customize is the annotation parser (parsers.py), and possibly the number of classses and their id.

## Training
If preprocessing was sucessfull, run train.py on your preprocessed data folder. Training can take a significant time even on high-end GPUs (~2-3 days on a Nvidia 2080Ti for the Lookit dataset.

## Testing
test.py can be used to test results

## Visualizing
visualize.py can help visualize and compare results, but it assumes a certain structure of the dataset and the output from test.py, try following through the results from Lookit if you wish to understand how this works.
