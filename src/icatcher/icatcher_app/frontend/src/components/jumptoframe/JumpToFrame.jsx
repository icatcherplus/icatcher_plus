import {
  Button,
  TextField
} from '@mui/material';
import React, { useEffect, useRef, useState } from 'react';
// import styles from './VideoHeader.module.css';
import { useSnacksDispatch, addSnack } from '../../state/SnacksProvider';
import { useVideoData } from '../../state/VideoDataProvider';
import { usePlaybackState } from '../../state/PlaybackStateProvider';


const styleOverrides = {
  /* style settings for jump to frame button */
  button: {
    borderColor: '#6b6b6b',
    color: '#6b6b6b',
//     height: 20,
    textTransform: 'none',
    padding: 0,
//     size: "medium",
    span: "none",
//     fullWidth: true,
  },
  textField: {
    height: 90,

  }
}
  
/* Expected props:
  currentFrameIndex: int
  handleJumpToFrame: callback
*/
function VideoHeader(props, {children}) {
  console.log(props)
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

  const handleKeyPress = (event) => {
    if(event.key === 'Enter'){
      console.log('enter press here! ')
      handleJumpToFrame(currentInput.current)
    }
  }

  return (
    <React.Fragment>
      <div >
        <TextField id="outlined-basic"
//                 error={!!errors.number}
          label="frame #"
          variant="outlined"
          margin='dense'
          inputProps={{
            inputMode: 'numeric',
            pattern: '[0-9]*',
//                     height: '1px',
            style: {
//                     &:invalid: {
//                       color: '#cc3014'
//                     },
              height: "1px",},
          }}
//                   sx={styleOverrides.textField}
          onChange={handleInputChange}
          onKeyPress={handleKeyPress}
        />

        <Button
          onClick={()=> {handleJumpToFrame(currentInput.current)}}
          variant="outlined"
          sx={styleOverrides.button}
          >
          Jump to Frame
        </Button>
      </div>

    </React.Fragment>
  );
}
  
export default VideoHeader;