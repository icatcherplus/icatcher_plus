import {
//   Button,
  TextField,
  InputAdornment,
  IconButton,
  Tooltip
} from '@mui/material';
import SendRoundedIcon from '@mui/icons-material/SendRounded';
import React, { useEffect, useRef, useState } from 'react';
import { useSnacksDispatch, addSnack } from '../../state/SnacksProvider';
import { useVideoData } from '../../state/VideoDataProvider';
import { usePlaybackState } from '../../state/PlaybackStateProvider';
import styles from './JumpToFrame.module.css'

// const styleOverrides = {
//   /* style settings for jump to frame button */
//   button: {
//     borderColor: '#e6e6e6',
//     color: '#e6e6e6',
//     fontSize: '12px',
//     minHeight: '25px',
//     textTransform: 'none',
//     padding: 0,
// //     size: "medium",
// //     span: "none",
// //     fullWidth: true,
//   },
//   textField: {
//     height: "1px",
//     borderColor: 'red',
//     color: 'white',
//     textAlign: 'center',
//   }
// }
  
/* Expected props:
  currentFrameIndex: int
  handleJumpToFrame: callback
*/
function JumpToFrame(props, {children}) {
//   console.log(props)
  const { handleJumpToFrame } = props;
  const videoData = useVideoData();
  const playbackState = usePlaybackState();
  const dispatchSnack = useSnacksDispatch();
//   const dispatchPlaybackState = usePlaybackStateDispatch();
//   const frameImages = useRef([]);

  const currentFramerate = useRef(0);
  const currentInput = useRef();
  const [ visible, setVisible ] = useState(false);
  const [ validInput, setValidInput ] = useState(true);
  const totalFrames =  videoData.metadata.numFrames;

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
    setValidInput(checkInputValidity(currentInput.current))
  }

  const handleKeyPress = (event) => {
    if(event.key === 'Enter'){
//       console.log('enter press here! ')
      handleJumpToFrame(currentInput.current)
    }
  }

  const checkInputValidity = (framenum) => {
    if (framenum <= totalFrames && framenum >= 0) {
      return true
    } else {
      return false
    }
  }

  return (
    <React.Fragment>
      <div className={styles.item}>
        <Tooltip
          title={validInput ? '' : `input must be an integer between 0 and ${totalFrames}`}
          placement='top'
          disableInteractive
        >
          <TextField
          error={validInput ? false : true}
            id='outlined-basic'
            label='frame #'
            variant='outlined'
  //           InputLabelProps={{ shrink: true, }} // removes placeholder
            margin='dense'
            inputProps={{
              inputMode: 'numeric',
              style: {
                height: '1px',
                color: 'white',
                textAlign: 'left',
                },
            }}
  //           sx={styleOverrides.textField}
            sx={{ '& .MuiInputLabel-root': {
                  // change input text color
                  color: '#e0e0e0',
                  },
                '& .css-o9k5xi-MuiInputBase-root-MuiOutlinedInput-root': {
                // remove padding on right side of adornment
                  paddingRight: '5px',
                  },
                '& .css-14s5rfu-MuiFormLabel-root-MuiInputLabel-root': {
                // adjust placeholder text position
                  position: 'absolute',
                  top: '-7px',
                  fontSize: '.9rem',
                },
                '& .css-1d3z3hw-MuiOutlinedInput-notchedOutline': {
                // change border color of textfield
                  borderColor: '#9e9e9e',
                }
               }}
            onChange={handleInputChange}
            onKeyPress={handleKeyPress}
            InputProps={{
              endAdornment: (
                <InputAdornment position='start'>
                  <Tooltip
                    title='jump to frame'
                    placement='top'
                    disableInteractive
                  >
                    <IconButton
                      aria-label="jump to frame"
                      onClick={()=> {handleJumpToFrame(currentInput.current)}}
                      edge="end"
                    >
                      <SendRoundedIcon
                      fontSize={'small'}
                      sx={{color: '#e0e0e0'}}
                      />
                    </IconButton>
                  </Tooltip>
                </InputAdornment>
              ),
            }}
          />
        </Tooltip>
      </div>

    </React.Fragment>
  );
}
  
export default JumpToFrame;