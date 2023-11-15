import { 
  Slider
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
    '&::before': {
      boxShadow: '0 2px 12px 0 rgba(0,0,0,0.4)',
    },
    '&:hover, &.Mui-focusVisible': {
      boxShadow: `0px 0px 0px 8px rgb(0 0 0 / 16%)`,
    },
    '&.Mui-active': {
      width: 15,
      height: 15,
    },
  },
  '& .MuiSlider-rail': {
    opacity: 0.28,
  }
}

function VideoScrubBar() {

  const videoData = useVideoData();
  const dispatchSnack = useSnacksDispatch();
  const playbackState = usePlaybackState();
  const dispatchPlaybackState = usePlaybackStateDispatch();

  const sliderValue = useRef(0);

  useEffect(() => {
    dispatchPlaybackState(updateFrameRange(playbackState.frameRange, videoData, (m,s) => dispatchSnack(addSnack(m,s))))
  }, [videoData.metadata])  // eslint-disable-line react-hooks/exhaustive-deps

  const handleSliderChange = (event, value, activeThumb) => {
    sliderValue.current = value
    dispatchPlaybackState({
      type: 'setCurrentFrame',
      currentFrame: value
    })
  }

  sliderValue.current = playbackState.currentFrame === undefined ? 0 : playbackState.currentFrame

  return (
    <Slider
      getAriaLabel={() => 'video scrub bar'}
      getAriaValueText={(v) => `frame ${v}`}
      size="small"
      value={sliderValue.current}
      min={0}
      step={1}
      max={videoData.metadata.numFrames - 1 || 0}
      onChange={(e,v,a) => handleSliderChange(e,v,a,videoData,dispatchSnack,dispatchPlaybackState)}
      sx={sliderStyling}
      valueLabelDisplay="auto"
    />
  );
}

export default VideoScrubBar;

