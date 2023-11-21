import React, { useEffect, useRef, useState } from 'react';
import styles from './FrameInfo.module.css';
import { useSnacksDispatch, addSnack } from '../../state/SnacksProvider';
import { useVideoData } from '../../state/VideoDataProvider';
import { usePlaybackState } from '../../state/PlaybackStateProvider';

  
/* Expected props:
  currentFrameIndex: int
  handleJumpToFrame: callback
*/
function FrameInfo(props, {children}) {

  const videoData = useVideoData();
  const playbackState = usePlaybackState();
  const dispatchSnack = useSnacksDispatch();

  const currentFramerate = useRef(0);
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


  return (
    <React.Fragment>
{/*       <div> */}
{/*       <div /> */}
{/*       { visible ? */}
          <div className={styles.box}>

            <div className={styles.vertical} >
              <div className={styles.boldDescriptor}>Frame Number</div>
              <div>{playbackState.currentFrame} </div>
            </div>

            <div className={styles.vertical} >
              <div className={styles.boldDescriptor}>Label</div>
              <div>{videoData.annotations.machineLabel[playbackState.currentFrame]} </div>
            </div>

            <div className={styles.vertical} >
              <div className={styles.boldDescriptor}>Confidence</div>
              <div>{videoData.annotations.confidence[playbackState.currentFrame]} </div>
            </div>

          </div>

{/*           : <div /> */}
{/*         } */}
{/*       <div /> */}
{/*        </div>  */}
    </React.Fragment>
  );
}
  
export default FrameInfo;