import { 
  ButtonGroup,
  IconButton,
  Switch,
  Tooltip
} from '@mui/material';
import {
  ChevronLeftRounded,
  ChevronRightRounded
} from '@mui/icons-material';
import { usePlaybackState, usePlaybackStateDispatch } from '../../state/PlaybackStateProvider';
import { useVideoData } from '../../state/VideoDataProvider';
import styles from './JumpButton.module.css';


const styleOverrides = {
  buttonGroup: {
    backgroundColor: '#eeeeee',
    margin: "0 1em auto"
  },
  button: {
    default: {
      margin: "0 auto"
    },
    toggled: {
      fillOpacity: .66,
      margin: "0 auto"

    }
  }
}

  
function JumpButton(props) {
  
  const { 
    handleJump
  } = props;

  const playbackState = usePlaybackState();
  const videoData = useVideoData();

  return (
      <ButtonGroup 
        className={styles.buttonGroup} 
        variant="contained"
        size="small"
        sx={styleOverrides.buttonGroup}
      >
        <Tooltip 
          title={`Jump to Previous`}
          placement="top"
          disableInteractive
        >
          <IconButton
              id="jumpBack"
              aria-label="jump to previous"
              sx={playbackState.currentFrame === 8? styleOverrides.button.toggled : styleOverrides.button.default}
              onClick={() => handleJump(false)}
            >
              <ChevronLeftRounded 
              // className={styles.icon} 
              />
            </IconButton>
          </Tooltip>
          <Tooltip 
            title={`Jump to Next`} 
            placement="top"
            disableInteractive
          >
            <IconButton
              id="jumpForward"
              aria-label="jump to next"
              sx={playbackState.currentFrame === videoData.metadata.numFrames-1? styleOverrides.button.toggled : styleOverrides.button.default}
              onClick={() => handleJump(true)}
            >
              <ChevronRightRounded 
              // className={styles.icon} 
              />
            </IconButton>
          </Tooltip>
        </ButtonGroup>
  );
}

export default JumpButton;