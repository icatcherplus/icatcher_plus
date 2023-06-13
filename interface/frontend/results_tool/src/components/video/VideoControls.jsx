import { 
  IconButton
} from '@mui/material';
import { 
  PlayArrow
} from '@mui/icons-material';
import { useEffect } from 'react';
import { useSnackDispatch } from '../../state/SnackContext';
import { useVideoData, useVideoDataDispatch } from '../../state/VideoDataContext';
import styles from './VideoControls.module.css';


  
/* Expected props:
tbd
*/
function VideoControls(props) {
  
  const { tbd } = props;
  const videoData = useVideoData();
  const dispatchVideoData = useVideoDataDispatch();
  const dispatchSnack = useSnackDispatch();

  const handlePlayClick = (e) => {
    console.log("click")
  }

  return (
    <div>
      <div className={styles.controlsBar}>
        <IconButton >
          <PlayArrow 
            color="primary"
            fontSize="medium"
            onClick={handlePlayClick}
          />
        </IconButton>
        
      </div>
    </div>
  );
}

export default VideoControls;