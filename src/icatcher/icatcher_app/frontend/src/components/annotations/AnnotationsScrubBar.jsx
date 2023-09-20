import { 
  Box,
  Slider,
  SliderThumb,
} from '@mui/material';
import { useEffect, useRef } from 'react';
import { useSnacksDispatch, addSnack } from '../../state/SnacksProvider';
import { useVideoData } from '../../state/VideoDataProvider';
import { usePlaybackState, usePlaybackStateDispatch, updateFrameRange } from '../../state/PlaybackStateProvider';

import styles from './AnnotationsScrubBar.module.css'

const sliderStyling = {
  color: 'green',
  height: 4,
  padding: 0,
  paddingBottom: 0.5,
  zIndex: 1,
  '& .MuiSlider-thumb': {
    width: 10,
    height: 10, 
    alignItems: 'flex-start',
    zIndex: 1,
    transition: '0.3s cubic-bezier(.47,1.64,.41,.8)',
    '&:before': {
      boxShadow: '0 2px 12px 0 rgba(0,0,0,0.4)',
    },
    '&:hover, &.Mui-focusVisible': {
      boxShadow: `0px 0px 0px 8px rgb(0 0 0 / 16%)`,
    },
  },
}

function AnnotationScrubTrackComponent(props) {
  const { children, ...other } = props;
  return (
    <SliderThumb {...other}>
      {children}
      <Box
        sx={{
          width: 3,
          height: 175,
          backgroundColor: 'green',
          borderRight: '1px solid gray',
          '&:before': {
            boxShadow: '0 2px 12px 0 rgba(0,0,0,0.4)',
          },
          '&:hover, &.Mui-focusVisible': {
            boxShadow: `0px 0px 0px 2px rgb(0 0 0 / 16%)`,
          },
          '&.Mui-active': {
            width: 20,
            height: 20,
          },
        }}
      >
      </Box>
    </SliderThumb>
    
  );
}


function AnnotationsScrubBar() {

  const videoData = useVideoData();
  const dispatchSnack = useSnacksDispatch();
  const playbackState = usePlaybackState();
  const dispatchPlaybackState = usePlaybackStateDispatch();

  const sliderValue = useRef(0);

  useEffect(() => {
    dispatchPlaybackState(updateFrameRange(playbackState.frameRange, videoData, (m,s) => dispatchSnack(addSnack(m,s))))
  }, [videoData.metadata])   // eslint-disable-line react-hooks/exhaustive-deps

  const handleSliderChange = (event, value) => {
    sliderValue.current = value
    dispatchPlaybackState({
      type: 'setCurrentFrame',
      currentFrame: value
    })
  }

  sliderValue.current = playbackState.currentFrame === undefined ? 0 : playbackState.currentFrame

  return (
    <div className={styles.spacer}>
      <Slider
        step={1}
        slots={{thumb: AnnotationScrubTrackComponent}}
        getAriaLabel={() => 'annotation scrub bar'}
        getAriaValueText={(v) => `frame ${v}`}
        size="small"
        value={sliderValue.current}
        min={videoData.metadata.frameOffset || 0}
        max={videoData.metadata.numFrames - 1 || 0}
        onChange={(e,v,a) => handleSliderChange(e,v,a,videoData,dispatchSnack,dispatchPlaybackState)}
        sx={{...sliderStyling, 
          width: playbackState.videoWidth-20
        }}
        valueLabelDisplay="auto"
      />
    </div>
  );
}

export default AnnotationsScrubBar;