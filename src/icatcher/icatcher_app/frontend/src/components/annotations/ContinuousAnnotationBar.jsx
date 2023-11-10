// import {
//   TextField
// } from '@mui/material';
import { useEffect, useState } from 'react';
// import { addSnack, useSnacksDispatch } from '../../state/SnacksProvider';
import { useVideoData } from '../../state/VideoDataProvider';
// import { usePlaybackState, usePlaybackStateDispatch } from '../../state/PlaybackStateProvider';
import AnnotationBar from './AnnotationsBar';

import styles from './ContinuousAnnotationBar.module.css'

/* Expected props:
  id:string --> key for annotation array in videoData.annotations
  palette:string --> key for color palette from colorPalettes object below
*/

const colorPalettes = {
  confidence: {
    '0%': '#FFFFFF',
    '100%': '#000000'
  },
  monochromeRed: {
    '0%': '#EEBAB4',
    '100%': '#F05039'
  },
  monochromeBlue: {
    '0%': '#7CA1CC',
    '100%': '#1F449C'
  },
  default: {
    '0%': '#FFFFFF',
    '100%': '#000000'
  }
}

function ContinuousAnnotationBar(props) {
  
  const { id } = props;
  const videoData = useVideoData();
//   const playbackState = usePlaybackState();
//   const dispatchPlaybackState = usePlaybackStateDispatch();
//   const dispatchSnack = useSnacksDispatch();

//   const [ threshold, setThreshold ] = useState(0.80);
  const [ range, setRange ] = useState({ min: 0, max: 1})
  const [ annotationArray, setAnnotationArray ] = useState(videoData.annotations[id].slice(videoData.metadata.frameOffset));


  useEffect (()=> {
    let [ newMin, newMax ] = getRange()
    setRange({min: newMin, max: newMax})
//     if(newMin > threshold || newMax < threshold) {
//       setThreshold((newMax-newMin)/2)
//     }
    setAnnotationArray(videoData.annotations[id].slice(videoData.metadata.frameOffset))
  }, [id, videoData.annotations])  // eslint-disable-line react-hooks/exhaustive-deps

  const computeColorArray = () => {
    let tempColorArray = []
    const palette = colorPalettes[id || 'default']
    annotationArray.forEach((a, i) => {
      let rgb = blendColors(hexToRgb(palette['0%']), hexToRgb(palette['100%']), normalizeValue(a, range))
      tempColorArray[i] = 'rgb(' + rgb.r + ', ' + rgb.g + ', ' + rgb.b + ')';
    })
    return [...tempColorArray]
  }
//
//   const jumpToNextInstance = (forward) => {
//     const condition = (e) => {
//       return e < threshold
//     }
//     let next = -1
//      if(forward === true) {
//       let arraySlice = videoData.annotations[id].slice(playbackState.currentFrame + 1)
//       next = arraySlice.findIndex((e) => condition(e))
//       if (next !== -1) { next = next + playbackState.currentFrame + 1}
//
//     } else {
//       let arraySlice = videoData.annotations[id].slice(0, playbackState.currentFrame)
//       next = arraySlice.findLastIndex((e) => condition(e))
//      }
//      if (next !== -1) {
//       dispatchPlaybackState({
//         type: 'setCurrentFrame',
//         currentFrame: next
//        })
//      }
//   }
//
//   const handleThresholdChange = (e) => {
//     let targetValue = e.target.value
//     if(targetValue < range.min) {
//       addSnack(`Value is below minimum ${id}. Defaulting to minimum ${range.min}`, 'info')
//       targetValue = range.min
//     }
//     else if (targetValue > range.max) {
//       dispatchSnack(addSnack(`Value is above maximum ${id}. Defaulting to maximum ${range.max}`, 'info'))
//       targetValue = range.max
//     }
//     setThreshold(targetValue)
//   }
  
  return (
    <div className={styles.temp}>
      <AnnotationBar
        id={id}
        getColorArray={computeColorArray}
      >
{/*         handleJump={jumpToNextInstance} */}
{/*       > */}
{/*         <TextField */}
{/*           className={styles.threshold} */}
{/*           id={`${id}-threshold-jumper`} */}
{/*           label={"Threshold"} */}
{/*           name="Name" */}
{/*           type="number" */}
{/*           inputProps={{ */}
{/*             min: range[0], */}
{/*             max: range[1], */}
{/*             step: 0.01 */}
{/*           }} */}
{/*           margin="dense" */}
{/*           multiline={false} */}
{/*           value={threshold} */}
{/*           onChange={handleThresholdChange} */}
{/*         /> */}
      </AnnotationBar>
    </div>
  );
}
  
export default ContinuousAnnotationBar;

const normalizeValue = (value, range) => {
  //get value normalized between minimum and maximum 
  return (value-range.min)/(range.max-range.min)
}

const getRange = () => {
  const min = 0
  const max = 1
  return [min, max]
}

const hexToRgb = (hex) => {
  return {
      r: parseInt(hex.substring(1, 3), 16),
      g: parseInt(hex.substring(3, 5), 16),
      b: parseInt(hex.substring(5, 7), 16)
  };
}

const blendColors = (colorA, colorB, weight) => {
  return {
      r: Math.floor(colorA.r * (1 - weight) + colorB.r * weight),
      g: Math.floor(colorA.g * (1 - weight) + colorB.g * weight),
      b: Math.floor(colorA.b * (1 - weight) + colorB.b * weight)
  };
}