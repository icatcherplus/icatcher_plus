# iCatcher+ UI

## Introduction

The iCatcher+ app is a tool that allows users to interact with iCatcher+ in the browser. The tool is designed to operate entirely locally and not upload any input files to a remote. The app comprises two functions: 

1. Video Preprocessing
2. Results Visualization

### Video Preprocessing

**Goal:** Allow users to crop input videos and annotate the first frame of a face to input into the iCatcher+ model pipeline.

**Features:**

* TBD

### Results Visualization

**Goal:** Allow users to visualize and correct iCatcher+ outputs.

**Features:**

* TBD

## For Developers

### Installing and Running

#### Prerequisites: 

* Install [Node.js](https://nodejs.org/en/download).

Once you have Node.js working in your command line, you can install the project dependencies by navigating to the `/interface` directory and running `npm install`. From there, you can start the app using `npm start`. Once the app is launched, you can find it in the browser at `localhost:5000` or `127.0.0.1:5000`.

### Development Guidelines

All development for the apps should operate under the strict constraint that user videos should not be uploaded to the internet at any point. Ideally, the whole app can function without internet access.

## For Users

### Installing and Running

TBD

A future development goal is to use the Node.js package [`pkg`](https://www.npmjs.com/package/pkg) to create executables for distrubution to users, allowing them to bypass more complex setup or the need to download prerequisite tools.

### Using the UI

TBD

### Outputs

TBD
