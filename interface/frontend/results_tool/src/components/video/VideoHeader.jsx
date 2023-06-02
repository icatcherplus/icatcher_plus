import { 
  Dialog,
  DialogTitle,
  DialogContent
} from '@mui/material';
import { useEffect, useRef } from 'react';
import { useSnackDispatch } from '../../state/SnackContext';
import { useVideoData, useVideoDataDispatch } from '../../state/VideoDataContext';

  
/* Expected props:
tbd
*/
function VideoCanvas(props, {children}) {
  
  const { currentFrameIndex } = props;
  const videoData = useVideoData();
  const dispatchSnack = useSnackDispatch();
  const smpteOffset = useRef(0);
  const currFramerate = useRef(0);

  useEffect(() => {
    if(Object.keys(videoData.metadata).length !== 0) {
      if (videoData.metadata.fps === undefined) {
        dispatchSnack({
          type: 'pushSnack',
          severity: 'error',
          message: 'No frames per second rate found, defaulting to 30.\nPlayback and timestamp accuracy will be affected.'
        });
        currFramerate.current = 30;
      } else { currFramerate.current = (videoData.metadata.fps); }

      if (videoData.metadata.smpteOffset === undefined) {
        dispatchSnack({
          type: 'pushSnack',
          severity: 'info',
          message: 'No timestamp offset found, defaulting to 0.\nTimestamps in browser may differ from original video.\nFrame numbers are still accurate.'
        });
        smpteOffset.current = 0;
      } else { smpteOffset.current = videoData.metadata.smpteOffset }
    } 
  },[videoData.metadata.fps])

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

  const smpteTime = smpteOffset.current + currentFrameIndex;
  const smpteString = getSMPTETime(currentFrameIndex, currFramerate)
  // const utcTime = utcOffset + ((1000/currFramerate.current)*currentFrameIndex);
  // const utcString = new Date(utcTime).toISOString()
  const label = 'tbd'
  const confidence = 'tbd'

  return (
    <div>
      {/* <div> UTC Time: {utcTime} </div> */}
      <div> SMPTE Time {smpteString}</div>
      <div> Frame Number: {currentFrameIndex} </div>
      <div> Label: {label}</div>
      <div> Confidence: {confidence}</div>
    </div>
  );
}
  
export default VideoCanvas;