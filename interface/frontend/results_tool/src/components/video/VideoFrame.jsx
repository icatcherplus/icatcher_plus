import React, { useEffect, useRef } from 'react';
import styles from './VideoFrame.module.css';

// import { useSnacksDispatch, addSnack } from '../../state/SnacksProvider';
import { useVideoData } from '../../state/VideoDataProvider';
import { usePlaybackState, usePlaybackStateDispatch, getNextFrame } from '../../state/PlaybackStateProvider';

import VideoScrubBar from './VideoScrubBar'
import VideoCanvas from './VideoCanvas';
import VideoControls from './VideoControls';
import AnnotationsFrame from '../annotations/AnnotationsFrame';
import VideoHeader from './VideoHeader';


function VideoFrame() {

  const videoData = useVideoData();
  const playbackState = usePlaybackState();
  const dispatchPlaybackState = usePlaybackStateDispatch();
  
  const playTimer = useRef(null);
  const startedLoad = useRef(false);
  const frameImages = useRef([]);
  const totalLoaded = useRef(0);
  const firstFrameIndex = useRef();
  
  useEffect(() => {

    if (videoData.frames.length !== 0 && Object.keys(videoData.metadata).length !== 0 && !startedLoad.current) {
      loadFrames();
      startedLoad.current = true;
      // expandMetadata(0); <-creates array of utc and smtpe timestamps for frames + optional cue frame info
    }
  }, [videoData.frames, videoData.metadata])

  const loadFrames = () => {
    console.log("videoData.frames.length", videoData.frames.length)
    videoData.frames.forEach((f, index) => {
      let img = new Image();
      img.frameNumber = f.frameNumber + videoData.metadata.frameOffset - 1;
      img.loaded = false;
      if (index === 0) {
        firstFrameIndex.current = img.frameNumber
      }
      
      img.onload = (e) => {
        let thisImg = e.currentTarget;
        thisImg.loaded = true;
        totalLoaded.current++;
        frameImages.current[thisImg.frameNumber] = thisImg;
        //add to state?
        if(thisImg.frameNumber === firstFrameIndex.current) {
          showFrame(thisImg.frameNumber);
          let firstImg = frameImages.current[thisImg.frameNumber]
          dispatchPlaybackState({
            type: 'setAspectRatio',
            aspectRatio: firstImg.width/firstImg.height
          })
        }
      }
      img.src = f.src;
    })
  }

  const showFrame = (index) => {
    pause();
    if ((typeof (frameImages.current[index]) === 'undefined') || (frameImages.current[index].loaded === false)) {
      return;
    }
    dispatchPlaybackState({
      type: 'setCurrentFrame',
      currentFrame: index
    })
  }
  
  const showNextFrame = () => {
    dispatchPlaybackState(getNextFrame(frameImages.current))
  }
  
  const pause = () => {
    dispatchPlaybackState({
      type: 'setPaused',
      paused: true
    })
  }

  const play = () => {
    if (videoData.frames.length === 0) {
      return;
    }
    if (playbackState.currentFrame === firstFrameIndex.current && playbackState.forwardPlay === false) {
      return;
    }
    if (playbackState.currentFrame === frameImages.current.length - 1 && playbackState.forwardPlay === true) {
      return;
    }
    if (playTimer.current === null) {
      playTimer.current = setInterval(
        showNextFrame, 
        playbackState.slowMotion 
          ? (1/videoData.metadata.framesPerSecond)*3000 
          : (1/videoData.metadata.framesPerSecond)*1000
      )
      dispatchPlaybackState({
        type: 'setPaused',
        paused: false
      })
    }
  }

  const togglePlay = () => {
    playTimer.current === null ? 
      play() : pause();
  }

  const toggleReverse = () => {
    dispatchPlaybackState({
      type: 'setForwardPlay',
      forwardPlay: !playbackState.forwardPlay
    })
    if (playbackState.paused === false) {
      clearInterval(playTimer.current)
      playTimer.current = playTimer.current = setInterval(
        showNextFrame, 
        playbackState.slowMotion
          ? (1/videoData.metadata.framesPerSecond)*3000
          : (1/videoData.metadata.framesPerSecond)*1000
      )
    }
  }

  const toggleSlowMotion = () => {
    if (playbackState.paused === false) {
      clearInterval(playTimer.current)
      playTimer.current = playTimer.current = setInterval(
        showNextFrame, 
        !playbackState.slowMotion
          ? (1/videoData.metadata.framesPerSecond)*3000 
          : (1/videoData.metadata.framesPerSecond)*1000
      )
    }
    dispatchPlaybackState({
      type: 'setSlowMotion',
      slowMotion: !playbackState.slowMotion
    })
  }

  const stepFrame = (forward) => {
    pause();
    //shift should jump to next labeled frame
    // if (!e.shiftKey) {
    showFrame(forward? playbackState.currentFrame + 1: playbackState.currentFrame - 1) 
    // }
  }

  const handleCanvasKeyDown = (e) => {
    let keyCode = e.keyCode;
    switch (keyCode) {
      case 32: { //Space
        togglePlay();
        break;
      }
      case 39: { //>
        stepFrame(true);
        break;
      }
      case 37: { //<
        stepFrame(false);
        break;
      }
      case 35: {
        pause();
        showFrame(frameImages.current.length - 1);
        break;
      }
      case 36: { //Home
        pause();
        showFrame(firstFrameIndex.current);
        break;
      }
      case 82: { //r
        toggleReverse();
        break;
      }
      case 83: { //s
        toggleSlowMotion();
        break;
      }
      default: {
        break;
      }
    }
  }

  if (playTimer.current != null && playbackState.paused === true) {
    clearInterval(playTimer.current)
    playTimer.current = null
  }

  return (
    <React.Fragment>
      <div className={styles.mainpage}>
        <div
          className={styles.videoFrame}
          style={{width: playbackState.videoWidth}}
        >
          <VideoHeader
            handleJumpToFrame={(i) => showFrame(Number(i))}
          />
          <VideoCanvas 
            className={styles.videoCanvas}
            frameToDraw={frameImages.current[playbackState.currentFrame]}
            handleClick={togglePlay}
            handleKeyDown={handleCanvasKeyDown}
          />
          <div 
            className={styles.controlsBox}
            style={{width: playbackState.videoWidth}}
          >
            <div 
              className={styles.controlsBackground}
            >
              <VideoScrubBar 
              />
              <VideoControls 
                togglePlay={togglePlay}
                toggleRev={toggleReverse}
                toggleSlowMotion={toggleSlowMotion} 
                stepBack={() => stepFrame(false)}
                stepForward={() => stepFrame(true)}
              />
            </div>
          </div>  
        </div>
        {/* <AnnotationsFrame /> */}
      </div>
    </React.Fragment>
  );
}

export default VideoFrame;