# iCatcher+ UI

## Introduction

The iCatcher+ app is a tool that allows users to interact with output from the iCatcher+ ML pipeline in the browser. The tool is designed to operate entirely locally and will not upload any input files to remote servers.

### Results Visualization

**Goal:** Allow users to visualize iCatcher+ outputs.

**Features:**

* Load a processed video into the browser for review—this will NOT upload to remote sources, the data will stay on the machine you are running the iCatcher+ UI on

* Review video annotations via annotated video playback and frame-by-frame navigation

* Jump to specific frame numbers

* Explore overview visualizations for gaze labels and confidence levels throughout the video

* Jump through label boundaries and low confidence frames to speed up the annotation process

## For Users

The install process is identical to the overall iCatcher+ quick install process. To review:

### User Prerequisites

1. If you haven't already, install [Python](https://www.python.org/downloads/). iCatcher+ requires Python version >= 3.9.
2. Install ffmpeg [ffmpeg](https://www.ffmpeg.org/) on your system. When using for the first time, neural network model files will automatically be downloaded to a local cache folder. To control where they are downloaded to set the "ICATCHER_DATA_DIR" environment variable.

### User Install

1. If you use Python for other projects, especially other ML projects, we strongly recommend using a python package manager like [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) to create a dedicated iCatcher+ virtual environment. See the main [iCatcher+ README](../../../README.md#step-2-create-a-conda-virtual-environment) for instructions on creating a conda enviroment. Remember to activate your environment before step 2.
2. Install iCatcher+ by running `pip install icatcher` in you local command line.

If you have already installed iCatcher+ and wish to update the package to include new functionality, run `pip install icatcher --upgrade` in your command line.

### Running the UI

To run the iCatcher+ UI, do the following:

1. Activate your virtual environment
2. In your command line, run `icatcher-app`

iCatcher+ should then pop up. If it doesn't, open your browser and go to [http://localhost:5001](http://localhost:5001)

### Using the UI

When you open the iCatcher+ UI, you will be met with a pop-up inviting you to upload your video directory. Please note, this requires you to upload *the whole output directory* which should include a `labels.txt` file, a `metadata.json` file, and a sub-directory containing all of the frame images from the video.

Once you've submitted the video, you should see a pop-up asking if you want to upload the whole video. Rest assured, this will not upload those files through the internet or to any remote servers. This is only giving the local browser permission to access those files. The files will stay local to whatever computer is running the browser.

At this point, you should see your video on the screen (you may need to give it a few second to load). Now you can start to review your annotations. Below the video you'll see heatmaps giving you a visual overview of the labels for each frame, as well as the confidence level for each frame.

### Outputs

For now, the iCatcher+ UI does not produce any outputs. Instead, we recommend you open the 'labels.txt' file that's output by the iCatcher+ video processing capability and edit by hand using the iCatcher+ UI to help you in your review.

The iCatcher+ team hopes to provide an annotation editing experience that's integrated into the UI in the near future.

## For Developers

### Prerequisites

1. Install Python >= 3.9
2. Install [Node.js](https://nodejs.org/en/download).

### Install

* Clone the git repository and check out a personal branch

* Install the full project by running `pip install -e .` in the main directory.

* Once you have Node.js working in your command line, you can install the frontend dependencies from the `/frontend` directory by running `npm ci` (preferred) or `npm install`.

### Running

To run the UI for testing or use, you can use the approach descibed above—run `icatcher-app` in your terminal. For development, we recommend one of two approaches:

1. For backend development, start the API in development mode: From this directory, run `python api/api.py`. Once the app is launched, you can find the Flask interactive UI in the browser at `http://localhost:5001/` or `127.0.0.1:5001`.

2. For frontend development, start the frontend code in development mode. To do so, navigate to the `frontend` directory and run `npm start`. Once launched, you can find this code at `http://localhost:3000/`. Running this way will allow the app to update views automatically when changes are saved to frontend files. This will serve only the code in the `frontend` directory; you may want to do step 1 concurrently so that your app can access the API.

### Development Guidelines

All development should operate under the strict constraint that user videos should not be uploaded to the internet at any point. Ideally, the whole app can function without internet access.
