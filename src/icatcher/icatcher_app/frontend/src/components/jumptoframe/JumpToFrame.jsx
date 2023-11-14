import {
  Button,
  TextField
} from '@mui/material';
import React, { useEffect, useRef, useState } from 'react';
// import styles from './VideoHeader.module.css';
import { useSnacksDispatch, addSnack } from '../../state/SnacksProvider';
import { useVideoData } from '../../state/VideoDataProvider';
import { usePlaybackState } from '../../state/PlaybackStateProvider';
import styles from './JumpToFrame.module.css'

const styleOverrides = {
  /* style settings for jump to frame button */
  button: {
    borderColor: '#e6e6e6',
    color: '#e6e6e6',
    fontSize: '12px',
    minHeight: '25px',
    textTransform: 'none',
    padding: 0,
//     size: "medium",
//     span: "none",
//     fullWidth: true,
  },
  textField: {
    height: "1px",
    borderColor: 'red',
    color: 'white',
    textAlign: 'center',
  }
}
  
/* Expected props:
  currentFrameIndex: int
  handleJumpToFrame: callback
*/
function JumpToFrame(props, {children}) {
  console.log(props)
  const { handleJumpToFrame } = props;
  const videoData = useVideoData();
  const playbackState = usePlaybackState();
  const dispatchSnack = useSnacksDispatch();
//   const dispatchPlaybackState = usePlaybackStateDispatch();
//   const frameImages = useRef([]);

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

//   const showFrame = (index) => {
//     pause();
//     if ((typeof (frameImages.current[index]) === 'undefined') || (frameImages.current[index].loaded === false)) {
//       return;
//     }
//     dispatchPlaybackState({
//       type: 'setCurrentFrame',
//       currentFrame: index
//     })
//   }
//   const handleJumpToFrame= (i) => showFrame(Number(i))
//
//   const pause = () => {
//     dispatchPlaybackState({
//       type: 'setPaused',
//       paused: true
//     })
//   }

  return (
    <React.Fragment>
      <div className={styles.item}>
        <TextField id="outlined-basic"
//                 error={!!errors.number}
          label="frame #"
          variant="outlined"
          margin='dense'
//           sx={{ input: { color: 'red' } }}
          inputProps={{
            inputMode: 'numeric',
//             pattern: '[0-9]*',
//                     height: '1px',
            style: {
              height: "1px",
//               borderColor: 'red',
              color: 'white',
              textAlign: 'center',
              },
//                 &:invalid: {
//                   color: '#cc3014'
//                 },
          }}
//           sx={styleOverrides.textField}
          sx={{ "& .MuiInputLabel-root": {
//               right: 0,
              textAlign: "center",
              color: "#e0e0e0",
             }}} // change placeholder text color and alignment
          onChange={handleInputChange}
          onKeyPress={handleKeyPress}
        />

        <Button
          onClick={()=> {handleJumpToFrame(currentInput.current)}}
//           onClick={()=> {showFrame(Number(currentInput.current))}}
          variant="outlined"
          sx={styleOverrides.button}
          >
          Jump to Frame
        </Button>
      </div>

    </React.Fragment>
  );
}
  
export default JumpToFrame;