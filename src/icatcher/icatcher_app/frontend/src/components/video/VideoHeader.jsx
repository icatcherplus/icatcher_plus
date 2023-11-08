import React, { useEffect, useRef, useState } from 'react';
import styles from './VideoHeader.module.css';
import { useSnacksDispatch, addSnack } from '../../state/SnacksProvider';
import { useVideoData } from '../../state/VideoDataProvider';
import { usePlaybackState } from '../../state/PlaybackStateProvider';


  
/* Expected props:
  currentFrameIndex: int
  handleJumpToFrame: callback
*/
function VideoHeader(props, {children}) {
  const { handleJumpToFrame } = props;
  const videoData = useVideoData();
  const playbackState = usePlaybackState();
  const dispatchSnack = useSnacksDispatch();

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
    } 
  },[videoData.metadata.fps])  // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (!visible && playbackState.currentFrame !== undefined) {
      setVisible(true)
    }
  }, [playbackState.currentFrame])  // eslint-disable-line react-hooks/exhaustive-deps

  const handleInputChange = (e) => {
    currentInput.current = Number(e.target.value);
  } 

  return (
    <React.Fragment>
      <div >
        <div />
        { visible ?
          <div
            className={styles.videoHeader}
            style={{width: playbackState.videoWidth}}
          >
            <div className={styles.vertical} > 
              <div>Frame Number:</div> 
              <div>{playbackState.currentFrame} </div>
            </div>
            <div className={styles.vertical} > 
              <div>Label:</div> 
              <div>{videoData.annotations.machineLabel[playbackState.currentFrame]} </div>
            </div>
            <div className={styles.vertical} > 
              <div>Confidence:</div> 
              <div>{videoData.annotations.confidence[playbackState.currentFrame]} </div>
            </div>
            <div className={styles.vertical} >
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