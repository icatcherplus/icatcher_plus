import os
import shutil
import subprocess
import logging
from icatcher.icatcher_app.api.api import REACT_BUILD_FOLDER, REACT_APP_FILE

logging.basicConfig(level = logging.WARNING)
logger = logging.getLogger(__name__)

def build_app(force=False, debug=False, info=True):
    if debug:
        logger.setLevel(logging.DEBUG)
    elif info:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.WARNING)

    if os.path.isfile(f'{REACT_BUILD_FOLDER}/{REACT_APP_FILE}'):
        if not force:
            logger.info(f'App buid already exists at {os.path.abspath(REACT_BUILD_FOLDER)}')
            return
        else:
            logger.info(f'Removing existing app buid at {os.path.abspath(REACT_BUILD_FOLDER)}')
            shutil.rmtree(REACT_BUILD_FOLDER)
    
    logger.info(f'Building iCatcher+ app at {os.path.abspath(REACT_BUILD_FOLDER)}')
    if shutil.which('npm') is None:
        raise Exception(
            '''
                Attempt to build app locally failed. Cannot build iCatcher app code locally without Node.js.\n

                Install iCatcher via `pip install icatcher` to obtain a pre-built version of the app code.
                To build locally, please install Node via https://nodejs.org/en and try again.
            '''
        )
    logger.info(f'Installing react app dependencies')
    subprocess.run(["npm ci"], cwd=f"{os.path.dirname(os.path.abspath(REACT_BUILD_FOLDER))}", shell=True)
    logger.info(f'Building react app')
    subprocess.run(["npm run build"], cwd=f"{os.path.dirname(os.path.abspath(REACT_BUILD_FOLDER))}", shell=True)



if __name__ == "__main__":
    build_app(force=False, debug=False, info=True)
