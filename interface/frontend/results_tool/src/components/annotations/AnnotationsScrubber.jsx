import { 
  Box,
  Slider,
  SliderRail,
  SliderThumb,
  SliderTrack
} from '@mui/material';
import { useEffect, useRef } from 'react';
import { useSnacksDispatch, addSnack } from '../../state/SnacksProvider';
import { useVideoData } from '../../state/VideoDataProvider';
import { usePlaybackState, usePlaybackStateDispatch, updateFrameRange } from '../../state/PlaybackStateProvider';


const sliderStyling = {
  color: 'red',
  height: 4,
  padding: 0,
  paddingBottom: 0.5,
  zIndex: 1,
  '& .MuiSlider-thumb': {
    width: 8,
    height: 8,
    transition: '0.3s cubic-bezier(.47,1.64,.41,.8)',
    '&:before': {
      boxShadow: '0 2px 12px 0 rgba(0,0,0,0.4)',
    },
    '&:hover, &.Mui-focusVisible': {
      boxShadow: `0px 0px 0px 8px rgb(0 0 0 / 16%)`,
    },
    '&.Mui-active': {
      width: 20,
      height: 20,
    },
  },
  '& .MuiSlider-rail': {
    opacity: 0.28,
  }
}

function AnnotationScrubTrackComponent(props) {
  const { children, ...other } = props;
  return (
    
      <SliderThumb {...other}>
        {children}
        <Box
        sx={{
          // width: 300,
          height: 300,
          backgroundColor: 'red',
          '&:hover': {
            backgroundColor: 'black',
            opacity: [0.9, 0.8, 0.7],
          },
          '&:before': {
            boxShadow: '0 2px 12px 0 rgba(0,0,0,0.4)',
          },
          '&:hover, &.Mui-focusVisible': {
            boxShadow: `0px 0px 0px 8px rgb(0 0 0 / 16%)`,
          },
          '&.Mui-active': {
            width: 20,
            height: 20,
          },
        }
      }></Box>
        {/* <span className="airbnb-bar" />
        <span className="airbnb-bar" />
        <span className="airbnb-bar" /> */}
      </SliderThumb>
    // </Box>
    
  );
}


function AnnotationsScrubber() {

  const videoData = useVideoData();
  const dispatchSnack = useSnacksDispatch();
  const playbackState = usePlaybackState();
  const dispatchPlaybackState = usePlaybackStateDispatch();

  const sliderValue = useRef(0);

  useEffect(() => { // move to app level ?
    dispatchPlaybackState(updateFrameRange(playbackState.frameRange, videoData, (m,s) => dispatchSnack(addSnack(m,s))))
  }, [videoData.metadata])

  const handleSliderChange = (event, value) => {
    sliderValue.current = value
    dispatchPlaybackState({
      type: 'setCurrentFrame',
      currentFrame: value
    })
  }

  sliderValue.current = playbackState.currentFrame === undefined ? 0 : playbackState.currentFrame

  return (
    <Slider
      slots={{thumb: AnnotationScrubTrackComponent}}
      getAriaLabel={() => 'annotation scrub bar'}
      getAriaValueText={(v) => `frame ${v}`}
      size="small"
      value={sliderValue.current}
      min={videoData.metadata.frameOffset || 0}
      step={1}
      max={videoData.metadata.numFrames - 1 || 0}
      onChange={(e,v,a) => handleSliderChange(e,v,a,videoData,dispatchSnack,dispatchPlaybackState)}
      sx={sliderStyling}
      valueLabelDisplay="auto"
    />
  );
}

export default AnnotationsScrubber;