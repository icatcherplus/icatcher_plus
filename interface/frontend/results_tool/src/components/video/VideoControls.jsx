import { 
  ButtonGroup,
  IconButton,
  Switch,
  Tooltip
} from '@mui/material';
import {
  FastForwardRounded,
  FastRewindRounded,
  PauseRounded,
  PlayArrowRounded,
  SkipNextRounded,
  SkipPreviousRounded,
  SlowMotionVideoRounded
} from '@mui/icons-material';
import styles from './VideoControls.module.css';

  
function VideoControls(props) {
  
  const { 
    togglePlay,
    pause, 
    toggleRev,
    toggleSlowMotion,    
    showFrame,
    currentFrame,
    isPlaying,
    isForward,
    isSlowMotion
  } = props;


  const styleOverrides = {
    button: {
      default: {
        color: 'white',
        borderRadius: 1,
        '.MuiTouchRipple-ripple .MuiTouchRipple-child': {
          borderRadius: 1,
          backgroundColor: 'lightgray',
        }
      },
      toggled: {
        color: 'white',
        fillOpacity: .66,
        borderRadius: 1,
        '.MuiTouchRipple-ripple .MuiTouchRipple-child': {
          borderRadius: 1,
        }
      }
    },
    switch: {
      '.MuiSwitch-switchBase': {
        margin: '8px',
        color: 'white',
        backgroundColor: 'gray',
        width: '20px',
        height: '20px',
        '&:hover': {
          backgroundColor: 'gray'
        },
        '& + .MuiSwitch-track': {
          backgroundColor: 'gray'
        },
        '&.Mui-checked': {
          color: 'gray',
          backgroundColor: 'white',
          '& + .MuiSwitch-track': {
            backgroundColor: 'white'
          },
          '&:hover': {
            backgroundColor: 'white'
          }
        }
      }
    }
  }

  const handlePlayPauseClick = (e) => {
    console.log("play/pause")
    togglePlay()
  }

  const handleStepBackClick = (e) => {
    pause();
    showFrame(currentFrame - 1)    
    console.log("set frame - 1")
  }

  const handleStepForwardClick = (e) => {
    pause();
    showFrame(currentFrame + 1)
    console.log("set frame + 1")
  }

  const handleReverseClick = (e) => {
    toggleRev();
    console.log("reverse reverse!")
  }

  const handleSlowMotionClick = (e) => {
    toggleSlowMotion();
    
    console.log("toggle slow motion")
  }


  return (
    <div className={styles.controlsBar}>
      <ButtonGroup className={styles.buttonGroup} >
        <Tooltip 
          title={isPlaying? "Pause (space)": "Play (Space)"} 
          placement="top" 
          disableInteractive
        >
          <IconButton
            id="playPause"
            aria-label="toggle play"
            sx={styleOverrides.button.default}
            onClick={handlePlayPauseClick}
          >
            { isPlaying
              ? <PauseRounded 
                  fontSize={'large'}
                  className={styles.icon}
                /> 
              : <PlayArrowRounded  
                  fontSize={'large'}
                  className={styles.icon}
                /> 
            }
          </IconButton>
        </Tooltip>
        <Tooltip 
          title={"Step Back (<)"} 
          placement="top"
          disableInteractive
        >
          <IconButton
            id="stepReverse"
            aria-label="step back one frame"
            sx={styleOverrides.button.default}
            onClick={handleStepForwardClick}
          >
            <SkipPreviousRounded 
              fontSize={'large'}
              className={styles.icon}
            />
          </IconButton>
        </Tooltip>
        <Tooltip 
          title={"Step Forward (>)"} 
          placement="top"
          disableInteractive
        >
          <IconButton
            id="stepForward"
            aria-label="step forward one frame"
            sx={styleOverrides.button.default}
            onClick={handleStepBackClick}
          >
            <SkipNextRounded 
              fontSize={'large'}
              className={styles.icon}
            />
          </IconButton>
        </Tooltip>
      </ButtonGroup>

      <ButtonGroup className={styles.buttonGroup} >
        <Tooltip 
          title={"Slow motion mode"} 
          placement="top"
          disableInteractive
        >
          <IconButton
              id="slowMotion"
              aria-label="toggle slow motion play"
              sx={isSlowMotion? styleOverrides.button.toggled : styleOverrides.button.default}
              onClick={handleSlowMotionClick}
            >
              <SlowMotionVideoRounded 
              className={styles.icon} 
              fontSize="large" 
              />
            </IconButton>
          </Tooltip>
          <Tooltip 
            title={`Reverse ${isForward ? "off": "on"} (r)`} 
            placement="top"
            disableInteractive
          >
            <Switch
              id="reverseSwitch"
              aria-label="toggle reversed playback"
              sx={styleOverrides.switch}
              icon={
                <FastForwardRounded fontSize="16px"/>
              }
              checkedIcon={
                <FastRewindRounded fontSize="16px"/>
            }
              checked={!isForward}
              onChange={handleReverseClick}
            >
            </Switch>
          </Tooltip>
        </ButtonGroup>
    </div>
  );
}

export default VideoControls;