import React, { useEffect, useRef, useState } from 'react';
import styles from './VideoFrame.module.css';
import { AspectRatio } from '@mui/joy';


import { useSnackDispatch } from '../../state/SnacksProvider';
import { useVideoData, useVideoDataDispatch } from '../../state/VideoDataProvider';
import ProgressBar from './ProgressBar'
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

  const [ currentFrame, setCurrentFrame ] = useState();
  const [ aspectRatio, setAspectRatio ] = useState(16/9);


  const startedLoad = useRef(false);
  const frameImages = useRef([]);
  const totalLoaded = useRef(0);
  const firstFrameIndex = useRef();
  const playState = useRef({
    forward: true,
    timer: null
  })
  
  useEffect(() => {

    if (videoData.frames.length !== 0 && Object.keys(videoData.metadata).length !== 0 && !startedLoad.current) {
      loadFrames();
      startedLoad.current = true;
      // expandMetadata(0); <-creates array of utc and smtpe timestamps for frames + optional cue frame info
    }
  }, [videoData.frames, videoData.metadata])

  // useEffect (()=> {
  //   if (playState.current.timer == null) {
  //     playState.current.forward = true;
  //   }
  // }, [playState.current.timer])

  const loadFrames = () => {
    videoData.frames.forEach((f, index) => {
      let img = new Image();
      img.frameNumber = f.frameNumber + videoData.metadata.frameOffset - 1;
      img.loaded = false;
      if (index === 0) {
        firstFrameIndex.current = img.frameNumber
        // console.log('hit load, setting first frame index', firstFrameIndex.current)
        // console.log(videoData.metadata.frameOffset)
      }
      
      img.onload = (e) => {
        let thisImg = e.currentTarget;
        thisImg.loaded = true;
        totalLoaded.current++;
        frameImages.current[thisImg.frameNumber] = thisImg;
        //add to state?
        if(thisImg.frameNumber === firstFrameIndex.current) {
          // console.log("showing first frame", thisImg.frameNumber)
          showFrame(thisImg.frameNumber);
        }
      }
      img.src = f.src;
    })
  }

  const showFrame = (index) => {
    // console.log('show frame', index)
    pause();
    setCurrentFrame((frame) => {
      if ((typeof (frameImages.current[index]) === 'undefined') || (frameImages.current[index].loaded === false)) {
        // console.log('bad', frameImages.current)

        return frame;
      }
      // console.log('update')
      return index;
    });
  }

  const showNextFrame = () => {
    setCurrentFrame((frame) => {
      let nextFrame = playState.current.forward ? frame + 1 : frame - 1;
      if ((typeof (frameImages.current[nextFrame]) === 'undefined') || (frameImages.current[nextFrame].loaded === false)) {
        pause();
        return frame;
      }
      return nextFrame;
    });

  }
  
  const pause = () => {
    // console.log('pause func')
    if (playState.current.timer !== null) {
      clearInterval(playState.current.timer)
      playState.current.timer = null;
    }
  }

  const play = (forward) => {
    if (videoData.frames.length === 0) {
      return;
    }
    playState.current.forward = forward;
    if (playState.current.timer === null) {
      playState.current.timer = setInterval(showNextFrame, (1/videoData.metadata.framesPerSecond)*1000)
    }
  }

  const togglePlay = (forward) => {
    playState.current.timer === null || playState.current.forward !== forward ? 
      play(forward) : pause();
  }

  const handleCanvasKeyDown = (e) => {
    // console.log("Key down:", e.keyCode)
    let keyCode = e.keyCode;
    switch (keyCode) {
      case 32: { //Space
        togglePlay(true);
        break;
      }
      case 39: { //>
        pause();
        //shift should jump to next labeled frame
        // if (!e.shiftKey) {
          showFrame(currentFrame + 1)
        // }
        break;
      }
      case 37: { //<
        pause();
        //shift should jump to previous labeled frame
        // if (!e.shiftKey) {
          showFrame(currentFrame - 1)
        // }
        break;
      }
      case 35: { //End (will stop playing automatically)
        showFrame(frameImages.current.length - 1);
        break;
      }
      case 36: { //Home
        pause();
        showFrame(firstFrameIndex.current);
        break;
      }
      case 82: { //R
        togglePlay(false);
        break;
      }
      default: {
        break;
      }
    }
  }

  return (
    <div className={styles.mainpage}>
      <div
        className={styles.videoFrame}
        // height={"854"}
        // width={"480"}
      >
        <VideoHeader
          currentFrameIndex={currentFrame}
          handleJumpToFrame={(i) => showFrame(Number(i))}
        />
        <VideoCanvas 
          className={styles.videoCanvas}
          frameToDraw={frameImages.current[currentFrame]}
          handleClick={() => togglePlay(true)}
          handleKeyDown={handleCanvasKeyDown}  
        />
        {/* <div className={styles.controlsBox}> */}
          <div className={styles.controlsBackground}>
            <ProgressBar />
            <VideoControls 
              togglePlay={togglePlay}
              pause={pause}
              toggleRev={()=>{playState.current.forward = !playState.current.forward}}
              toggleSlowMotion={()=> {console.log('todo: implement toggle slow-mo')}} 
              showFrame={showFrame}
              currentFrame={currentFrame}
              isPlaying={playState.current.timer != null}
              isForward={playState.current.forward === true}
              isSlowMotion={false}
            />
          </div>
        {/* </div> */}
      </div>
    </div>
  );
}

export default VideoFrame;