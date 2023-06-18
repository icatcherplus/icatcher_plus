import { 
  Slider
} from '@mui/material';
import { useState } from 'react';
import { useSnacksDispatch } from '../../state/SnacksProvider';
import { useVideoData, useVideoDataDispatch } from '../../state/VideoDataProvider';

function ProgressBar(props, {children}) {

  const { currentFrame } = props;
  const videoData = useVideoData();
  const dispatchVideoData = useVideoDataDispatch();
  const dispatchSnack = useSnacksDispatch();

  const [ frameRange, setFrameRange ] = useState([0, 100]);

  const sliderStyling = {
    color: 'red',
    height: 4,
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

  const handleSliderChange = (e, value, activeThumb) => {
    setFrameRange(value)
    console.log(`Slider change: ${value}, ${activeThumb}`)
  }

  return (
    <Slider
      aria-label="video scrub bar"
      size="small"
      value={frameRange}
      min={videoData.metadata.frameOffset}
      step={1}
      max={videoData.metadata.numFrames}
      onChange={handleSliderChange}
      sx={sliderStyling}
    />
  );
}

export default ProgressBar;