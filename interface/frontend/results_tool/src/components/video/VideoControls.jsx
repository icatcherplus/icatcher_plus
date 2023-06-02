import { 
  Dialog,
  DialogTitle,
  DialogContent
} from '@mui/material';
import { useEffect } from 'react';
import { useSnackDispatch } from '../../state/SnackContext';
import { useVideoData, useVideoDataDispatch } from '../../state/VideoDataContext';

  
/* Expected props:
tbd
*/
function VideoControls(props) {
  
  const { tbd } = props;
  const videoData = useVideoData();
  const dispatchVideoData = useVideoDataDispatch();
  const dispatchSnack = useSnackDispatch();


  return (
    <div>
      <div id="videoControlBar" width="854" height="50">
      </div>
    </div>
  );
}

export default VideoControls;