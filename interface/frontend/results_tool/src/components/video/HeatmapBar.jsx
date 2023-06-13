import { 
  Dialog,
  DialogTitle,
  DialogContent
} from '@mui/material';
import { useEffect, useState } from 'react';
import { useSnackDispatch } from '../../state/SnackContext';
import { useVideoData, useVideoDataDispatch } from '../../state/VideoDataContext';

  
/* Expected props:
  tbd
*/
function HeatmapBar(props) {

  const { tbd } = props;
  const videoData = useVideoData();
  const dispatchVideoData = useVideoDataDispatch();
  const dispatchSnack = useSnackDispatch();
  const [ width, setWidth ] = useState(0);

  useEffect(() => {
    let loadedFrames = videoData.frames.length;
    if (loadedFrames > 0) {
      if (videoData.metadata.numFrames <= 0) {
        dispatchSnack({
          type:'pushSnack',
          severity:'warning',
          message:'Video metadata incorrectly lists video as 0 frames long'
        })
      }
      else {
        let w = (loadedFrames - videoData.frameOffset - 1)/videoData.metadata.numFrames * 100.0
        if (w > width) {
          setWidth(w)
        }
      }
    }
  }, [
    videoData.frames.length, 
    videoData.frameOffset,
    videoData.metadata.numFrames
  ])


  return (
    <div>
      <div id="downloadProgressBar">
        <div id="myBar"></div>
        <div id="myPos"></div>
        <div id="myFrameTC"></div>
        <div id="myFrameInfo"></div>
      </div>
    </div>
  );
}

export default HeatmapBar;