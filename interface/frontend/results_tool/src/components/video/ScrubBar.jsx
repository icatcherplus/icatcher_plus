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
  function VideoCanvas(props, {children}) {
    
    const { currentFrame } = props;
    const videoData = useVideoData();
    const dispatchVideoData = useVideoDataDispatch();
    const dispatchSnack = useSnackDispatch();
  
    useEffect(() => {
        // var elem = document.getElementById('myPos');
        // var w = document.getElementById('myFrameDownloadProgress').offsetWidth;
    
        // //Calculate x pos from current frame
        // var offsetxPos = (m_play_state.current_frame / (m_manifest.video.num_frames - 1)) * w;
    
        // elem.style.left = offsetxPos;
    }, [currentFrame])
  
    return (
      <div>
        {children}
      </div>
    );
  }
    
  export default VideoCanvas;