import { 
  Dialog,
  DialogTitle,
  DialogContent
} from '@mui/material';
import React, { useEffect, useRef, useState } from 'react';
import styles from './VideoFrame.module.css';

import { useSnackDispatch } from '../../state/SnackContext';
import { useVideoData, useVideoDataDispatch } from '../../state/VideoDataContext';
import KeystrokeMonitor from './KeystrokeMonitor';
import VideoCanvas from './VideoCanvas';
import VideoControls from './VideoControls';
import HeatmapBar from './HeatmapBar';
import ScrubBar from './ScrubBar';
import VideoHeader from './VideoHeader';

  
/* Expected props:
  tbd
*/
function VideoFrame(props) {

  const { tbd } = props;
  const videoData = useVideoData();
  const dispatchVideoData = useVideoDataDispatch();
  const dispatchSnack = useSnackDispatch();

  const startedLoad = useRef(false);
  const frameImages = useRef([]);
  const totalLoaded = useRef(0);
  const [ currentFrame, setCurrentFrame ] = useState();
  const playState = useRef({
    currentFrame: -1,
    timer: null,
    downloadedFrames: null
  })
  
  useEffect(() => {
    if (Object.keys(videoData.frames).length !== 0 && !startedLoad.current) {
      loadFrames();
      startedLoad.current = true;
      // expandMetadata(0); <-creates array of utc and smtpe timestamps for frames + optional cue frame info
    }
  }, [videoData.frames])

  const loadFrames = () => {
    videoData.frames.forEach((f, index) => {
      let img = new Image();
      img.frameNumber = f.frameNumber;
      img.loaded = false;
      img.isFirstFrame = (index === 0);
      img.onload = (e) => {
        let thisImg = e.currentTarget;
        thisImg.loaded = true;
        totalLoaded.current++;
        frameImages.current[thisImg.frameNumber] = thisImg;
        //add to state?
        if(thisImg.isFirstFrame) {
          // get width and height, pass to canvas
          showFrame(thisImg.frameNumber);
        }
      }
      img.src = f.src;
    })
  }

  const showFrame = (index) => {
    if (videoData.frames.length === 0) {
      console.log('NO FRAMES')
      return;
    }
    if ( (index <= 0) || (index >= frameImages.current.length)) {
      console.log('BAD FRAMES')
      pause();
      return;
    }
    if ((typeof (frameImages.current[index]) === 'undefined') || (frameImages.current.loaded === false)) {
      console.log('UNLOADED FRAMES')
      pause();
      return;
    }
    console.log("setting current frame to ", index)
    setCurrentFrame(index)
  }
  
  const pause = () => {
    if(playState.current.timer != null){
      clearInterval(playState.current.timer)
      playState.current = {
        ...playState.current,
        timer: null
      }
    }
  }

  return (
    <React.Fragment>
      {/* <KeystrokeMonitor> */}
        <div className={styles.videoFrame}>
          <VideoHeader currentFrameIndex={currentFrame}/>
          <VideoCanvas frameToDraw={frameImages.current[currentFrame]}/>
          <VideoControls />
        </div>
        {/* <ScrubBar>
          <HeatmapBar id="editsMap"/>
          <HeatmapBar id="labelsMap"/>
          <HeatmapBar id="confidenceMap"/>
        </ScrubBar> */}
      {/* </KeystrokeMonitor> */}
    </React.Fragment>
  );
}

export default VideoFrame;