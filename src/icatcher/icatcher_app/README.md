# iCatcher+ UI

For more information on the app, see [Web App](https://github.com/icatcherplus/icatcher_plus/tree/master#web-app) in the main iCatcher+ repository.

## For Developers

### Prerequisites

To run iCatcher+, you will need:
* [Python >=3.9](https://www.python.org/)
* [Node.js](https://nodejs.org/en/download)
* A virtual environment manager, such as [Miniconda](https://conda.io) or [virtualenv](https://pypi.org/project/virtualenv/)
* [Ffmpeg](https://www.ffmpeg.org/) installed in your system and available (if you used conda, you can quickly install it with `conda install -c conda-forge ffmpeg`)

### Install

* Clone the git repository

* Install the icatcher python module in editable mode by running `pip install -e .` in the main directory.

* Once you have Node.js working in your command line, you can install the frontend dependencies from the `/frontend` directory by running `npm ci` (preferred) or `npm install`.

### Running

To run the UI for testing or use, run `icatcher --app` in your terminal and it should open automatically at [http://localhost:5001](http://localhost:5001). If it's your first time running it, the launch will take a bit longer as the app is built into the `frontend/build` directory.

To run the UI for frontend development, navigate to the `frontend` directory and run `npm start`. Once launched, you can find this code at `http://localhost:3000/`. Running this way will allow the app to update views automatically when changes are saved to frontend files.

### Development Guidelines

All development should operate under the strict constraint that user videos should not be uploaded to the internet at any point. Ideally, the whole app can function without internet access.

### Troubleshooting

If you see a `404` error when trying to open the app at [http://localhost:5001](http://localhost:5001), try the following: 
* Double check that you have node insalled and the `npm` keyword available in your terminal, then try running again
* Delete the `src/icatcher/icatcher_app/frontend/build` directory and try running again
* If the following approaches don't work, you can build the app yourself by running `npm run build` in the `frontend` directory and try running `icatcher --app` again
