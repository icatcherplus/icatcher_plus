import React, { useEffect, useRef, useState } from 'react';
import styles from './VideoHeader.module.css';
import { useSnacksDispatch, addSnack } from '../../state/SnacksProvider';
import { useVideoData } from '../../state/VideoDataProvider';

  
/* Expected props:
currentFrameIndex: int
handleJumpToFrame: callback
*/
function VideoHeader(props, {children}) {
  
  const { currentFrameIndex, handleJumpToFrame, width } = props;
  const videoData = useVideoData();
  const dispatchSnack = useSnacksDispatch();
  const smpteOffset = useRef(0);
  const currFramerate = useRef(0);
  const currInput = useRef();
  const [ visible, setVisible ] = useState(false)

  useEffect(() => {
    if(Object.keys(videoData.metadata).length !== 0) {
      if (videoData.metadata.fps === undefined) {
        console.log('dispatching snack fps')
        addSnack(
          'No frames per second rate found, defaulting to 30.\nPlayback and timestamp accuracy will be affected.',
          'error',
          dispatchSnack
        )
        currFramerate.current = 30;
      } else { currFramerate.current = (videoData.metadata.fps); }

      // if (videoData.metadata.smpteOffset === undefined) {
        // addSnack(
        //   'No timestamp offset found, defaulting to 0.\nTimestamps in browser may differ from original video.\nFrame numbers are still accurate.',
        //   'info',
        //   dispatchSnack
        // )
      //   smpteOffset.current = 0;
      // } else { smpteOffset.current = videoData.metadata.smpteOffset }
    } 
  },[videoData.metadata.fps])

  useEffect(() => {
    if (!visible && currentFrameIndex !== undefined) {
      setVisible(true)
    }
  }, [currentFrameIndex])

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
    console.log('input', e.target.value)
    currInput.current = Number(e.target.value);
  } 

  const smpteTime = smpteOffset.current + currentFrameIndex;
  const smpteString = getSMPTETime(currentFrameIndex, currFramerate)
  // const utcTime = utcOffset + ((1000/currFramerate.current)*currentFrameIndex);
  // const utcString = new Date(utcTime).toISOString()
  const label = 'tbd'
  const confidence = 'tbd'

  return (
    <React.Fragment>
      <div >
        <div />
        { visible ?
          <div
            className={styles.videoHeader}
            style={{width: width}}
          >
            {/* <div> UTC Time: {utcTime} </div> */}
            {/* <div> SMPTE Time {smpteString}</div> */}
            <div> Frame Number: {currentFrameIndex} </div>
            <div> Label: {videoData.annotations[currentFrameIndex - 4]?.machineLabel}</div>
            <div> Confidence: {videoData.annotations[currentFrameIndex - 4]?.confidence}</div>
            <div>
              <input type='text' onChange={handleInputChange}></input>
              <button onClick={()=> {handleJumpToFrame(currInput.current)}}> Jump to Frame</button>
            </div>
          </div>
        :
          <div 
            className={styles.videoHeader} 
            style={{width: width}}
          />
        }
        <div />
      </div>
    </React.Fragment>
  );
}
  
export default VideoHeader;