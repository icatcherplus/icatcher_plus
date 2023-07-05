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
import { usePlaybackState, usePlaybackStateDispatch } from '../../state/PlaybackStateProvider';

import styles from './VideoControls.module.css';

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

  
function VideoControls(props) {
  
  const { 
    togglePlay,
    toggleRev,
    toggleSlowMotion,    
    stepBack,
    stepForward
  } = props;

  const playbackState = usePlaybackState();

  return (
    <div className={styles.controlsBar}>
      <ButtonGroup className={styles.buttonGroup} >
        <Tooltip 
          title={playbackState.paused? "Play (Space)" : "Pause (space)"} 
          placement="top" 
          disableInteractive
        >
          <IconButton
            id="playPause"
            aria-label="toggle play"
            sx={styleOverrides.button.default}
            onClick={togglePlay}
          >
            { playbackState.paused
              ? <PlayArrowRounded  
                  fontSize={'large'}
                  className={styles.icon}
                />
              : <PauseRounded 
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
            onClick={stepBack}
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
            onClick={stepForward}
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
          title={`Slow motion ${playbackState.slowMotion ? "on":"off"} (s)`} 
          placement="top"
          disableInteractive
        >
          <IconButton
              id="slowMotion"
              aria-label="toggle slow motion play"
              sx={playbackState.slowMotion? styleOverrides.button.default : styleOverrides.button.toggled}
              onClick={toggleSlowMotion}
            >
              <SlowMotionVideoRounded 
              className={styles.icon} 
              fontSize="large" 
              />
            </IconButton>
          </Tooltip>
          <Tooltip 
            title={`Reverse ${playbackState.forwardPlay ? "off":"on"} (r)`} 
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
              checked={!playbackState.forwardPlay}
              onChange={toggleRev}
            >
            </Switch>
          </Tooltip>
        </ButtonGroup>
    </div>
  );
}

export default VideoControls;