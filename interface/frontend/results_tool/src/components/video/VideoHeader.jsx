import React, { useEffect, useRef, useState } from 'react';
import styles from './VideoHeader.module.css';
import { useSnacksDispatch, addSnack } from '../../state/SnacksProvider';
import { useVideoData } from '../../state/VideoDataProvider';
import { usePlaybackState, usePlaybackStateDispatch } from '../../state/PlaybackStateProvider';


  
/* Expected props:
currentFrameIndex: int
handleJumpToFrame: callback
*/
function VideoHeader(props, {children}) {
  const { handleJumpToFrame } = props;
  const videoData = useVideoData();
  const playbackState = usePlaybackState();
  const dispatchSnack = useSnacksDispatch();

  const smpteOffset = useRef(0);
  const currentFramerate = useRef(0);
  const currentInput = useRef();
  const [ visible, setVisible ] = useState(false)

  useEffect(() => {
    if(Object.keys(videoData.metadata).length !== 0) {
      if (videoData.metadata.fps === undefined) {
        dispatchSnack(addSnack(
          'No frames per second rate found, defaulting to 30.\nPlayback accuracy will be affected.',
          'error'
        ))
        currentFramerate.current = 30;
      } else { currentFramerate.current = (videoData.metadata.fps); }

      // if (videoData.metadata.smpteOffset === undefined) {
        // dispatchSnack(addSnack(
        //   'No timestamp offset found, defaulting to 0.\nTimestamps in browser may differ from original video.\nFrame numbers are still accurate.',
        //   'info'
        // ))
      //   smpteOffset.current = 0;
      // } else { smpteOffset.current = videoData.metadata.smpteOffset }
    } 
  },[videoData.metadata.fps])

  useEffect(() => {
    if (!visible && playbackState.currentFrame !== undefined) {
      setVisible(true)
    }
  }, [playbackState.currentFrame])

  const getSMPTETime = (index, fps) => {

    let h = Math.floor(index / (fps * 3600));
    let remaining = index - (h * (fps * 3600));

    let m = Math.floor(remaining / (fps * 60));
    remaining = remaining - (m * (fps * 60));

    let s = Math.floor(remaining / fps);
    remaining = remaining - (s * fps);

    let f = remaining;

    return toTwoDigits(h) + ":" + toTwoDigits(m) + ":" +  toTwoDigits(s) + ":" + toTwoDigits(f);
  }

  const toTwoDigits = (s) => {
    let longS = s.toString();
    while (longS.length < 2) {
      longS = "0" + longS;
    }
    return longS;
  }

  const handleInputChange = (e) => {
    currentInput.current = Number(e.target.value);
  } 

  const smpteTime = smpteOffset.current + playbackState.currentFrame;
  const smpteString = getSMPTETime(playbackState.currentFrame, currentFramerate)
  // const utcTime = utcOffset + ((1000/currentFramerate.current)*currentFrameIndex);
  // const utcString = new Date(utcTime).toISOString()

  return (
    <React.Fragment>
      <div >
        <div />
        { visible ?
          <div
            className={styles.videoHeader}
            style={{width: playbackState.videoWidth}}
          >
            {/* <div> UTC Time: {utcTime} </div> */}
            {/* <div> SMPTE Time {smpteString}</div> */}
            <div> Frame Number: {playbackState.currentFrame} </div>
            <div> Label: {videoData.annotations.machineLabel[playbackState.currentFrame]}</div>
            <div> Confidence: {videoData.annotations.confidence[playbackState.currentFrame]}</div>
            <div>
              <input type='text' onChange={handleInputChange}></input>
              <button onClick={()=> {handleJumpToFrame(currentInput.current)}}> Jump to Frame</button>
            </div>
          </div>
        :
          <div 
            className={styles.videoHeader} 
            style={{width: playbackState.videoWidth}}
          />
        }
        <div />
      </div>
    </React.Fragment>
  );
}
  
export default VideoHeader;