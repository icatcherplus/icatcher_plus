import React, { useEffect, useRef, useState } from 'react';
import styles from './VideoFrame.module.css';

// import { useSnacksDispatch, addSnack } from '../../state/SnacksProvider';
import { useVideoData, useVideoDataDispatch } from '../../state/VideoDataProvider';
import VideoScrubBar from './VideoScrubBar'
import VideoCanvas from './VideoCanvas';
import VideoControls from './VideoControls';
import AnnotationsFrame from '../annotations/AnnotationsFrame';
import VideoHeader from './VideoHeader';

  
/* Expected props:
  tbd
*/
function VideoFrame(props) {

  const { tbd } = props;
  const videoData = useVideoData();
  // const dispatchSnack = useSnacksDispatch();

  const [ playState, setPlayState ] = useState({
    currentFrame: null,
    forwardPlay: true,
    paused: true,
    aspectRatio: 16/9,
    slowMotion: false
  })

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
          setPlayState(p => {
            return {
              ...p,
              aspectRatio: firstImg.width/firstImg.height
            }
          })
        }
      }
      img.src = f.src;
    })
  }

  const showFrame = (index) => {
    pause();
    setPlayState((p) => {
      if ((typeof (frameImages.current[index]) === 'undefined') || (frameImages.current[index].loaded === false)) {
        return p;
      }
      return { ...p, currentFrame: index};
    });
  }
  
  const showNextFrame = () => {
    setPlayState((p) => {
      console.log('interval run')
      let nextFrame = p.forwardPlay ? p.currentFrame + 1 : p.currentFrame - 1;
      if ((typeof (frameImages.current[nextFrame]) === 'undefined') || (frameImages.current[nextFrame].loaded === false)) {
        return { ...p, paused: true };
      }
      return { ...p, currentFrame: nextFrame };
    });
  }
  
  const pause = () => {
    setPlayState(p => {
      return {
        ...p, 
        paused: true
      }
    })
  }

  const play = () => {
    if (videoData.frames.length === 0) {
      return;
    }
    if (playState.currentFrame === firstFrameIndex.current && playState.forwardPlay === false) {
      return;
    }
    if (playState.currentFrame === frameImages.current.length - 1 && playState.forwardPlay === true) {
      return;
    }
    if (playTimer.current === null) {
      playTimer.current = setInterval(
        showNextFrame, 
        playState.slowMotion 
        ? (1/videoData.metadata.framesPerSecond)*3000 
        : (1/videoData.metadata.framesPerSecond)*1000
      )
      setPlayState((p) => {
        return {
          ...p,
          paused: false 
        }
      })
    }
  }

  const togglePlay = () => {
    playTimer.current === null ? 
      play() : pause();
  }

  const toggleReverse = () => {
    setPlayState(p => {
      return {
        ...p,
        forwardPlay: !p.forwardPlay
      }
    })
  }

  const toggleSlowMotion = () => {
    if (playState.paused === false) {
      let tempSlowMotion = playState.slowMotion;
      clearInterval(playTimer.current)
      playTimer.current = playTimer.current = setInterval(
        showNextFrame, 
        !tempSlowMotion 
        ? (1/videoData.metadata.framesPerSecond)*3000 
        : (1/videoData.metadata.framesPerSecond)*1000
      )
    }
    setPlayState((p) => {
      return {
        ...p,
        slowMotion: !p.slowMotion
      }
    })
  }

  const handleCanvasKeyDown = (e) => {
    console.log('key down')
    let keyCode = e.keyCode;
    switch (keyCode) {
      case 32: { //Space
        togglePlay();
        break;
      }
      case 39: { //>
        pause();
        //shift should jump to next labeled frame
        // if (!e.shiftKey) {
          showFrame(playState.currentFrame + 1)
        // }
        break;
      }
      case 37: { //<
        pause();
        //shift should jump to previous labeled frame
        // if (!e.shiftKey) {
          showFrame(playState.currentFrame - 1)
        // }
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

  const getWidth = () => {
    let videoWidth = (window.innerHeight * .6) * playState.aspectRatio
    if (videoWidth > (0.8 * window.innerWidth)) {
      videoWidth = (0.8 * window.innerWidth)
    }
    return videoWidth
  }

  let width = getWidth();

  if (playTimer.current != null && playState.paused === true) {
    clearInterval(playTimer.current)
    playTimer.current = null
  }

  return (
    <React.Fragment>
      <div className={styles.mainpage}>
        <div
          className={styles.videoFrame}
          style={{width: width}}
        >
          <VideoHeader
            currentFrameIndex={playState.currentFrame}
            handleJumpToFrame={(i) => showFrame(Number(i))}
            width={width}
          />
          <VideoCanvas 
            className={styles.videoCanvas}
            frameToDraw={frameImages.current[playState.currentFrame]}
            handleClick={togglePlay}
            handleKeyDown={handleCanvasKeyDown}
            width={width}
            aspectRatio={playState.aspectRatio}
          />
          <div 
            className={styles.controlsBox}
            style={{width: width}}
          >
            <div 
              className={styles.controlsBackground}
            >
              <VideoScrubBar currentFrame={playState.currentFrame}/>
              <VideoControls 
                togglePlay={togglePlay}
                pause={pause}
                toggleRev={toggleReverse}
                toggleSlowMotion={toggleSlowMotion} 
                showFrame={showFrame}
                currentFrame={playState.currentFrame}
                isPlaying={playState.paused === false}
                isForward={playState.forwardPlay}
                isSlowMotion={playState.slowMotion}
                width={width}
              />
            </div>
          </div>  
        </div>
        <AnnotationsFrame width={width} />

      </div>
    </React.Fragment>
  );
}

export default VideoFrame;