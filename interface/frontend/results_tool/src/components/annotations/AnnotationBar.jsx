import { useEffect, useState } from 'react';
import { addSnack, useSnacksDispatch } from '../../state/SnacksProvider';
import { useVideoData, useVideoDataDispatch } from '../../state/VideoDataProvider';
import { usePlaybackState, usePlaybackStateDispatch } from '../../state/PlaybackStateProvider';
import HeatmapCanvas from './HeatmapCanvas';
import JumpButton from '../common/JumpButton';

import styles from './AnnotationsBar.module.css'


/* Expected props:
tbd
*/

const colorPalettes = {
  categorical: {
    machineLabel: {
      'undefined': '#C5C5C5',
      'left': '#F05039',
      'right': '#1F449C',
      'away': '#EEBAB4',
      'noface': '#7CA1CC'
    },
    edited: []
  },
  continuous: {
    confidence: {
      '0%': '#FFFFFF',
      '100%': '#000000'
    },
    default: {
      '0%': '#FFFFFF',
      '100%': '#000000'
    }
  },
}

function AnnotationBar(props) {
  
  const { id, type } = props;
  const videoData = useVideoData();
  const playbackState = usePlaybackState();
  const dispatchPlaybackState = usePlaybackStateDispatch();

  const [ colorArray, setColorArray ] = useState([]);
  const [ colorPalette, setColorPalette ] = useState(id);

  useEffect(()=> {
    videoData.annotations[id].length > 0 ?
      computeColorArray()
    :
      setColorArray([]);

  },[videoData.annotations, id])

  const computeColorArray = () => {
    let tempColorArray = []
    const palette = colorPalettes[type][colorPalette || 'default']
    const startArray = videoData.annotations[id].slice(videoData.metadata.frameOffset-1)
    if (type === 'categorical') {
      // let categories = [ ...new Set(startArray)]
      // let colorMap = {}
      // categories.forEach((l, i) => {
      //   colorMap[l] = palette[i]
      // })
      // console.log(id, categories, colorMap)
      startArray.forEach((a, i) => {
      // videoData.annotations[id].forEach((a, i) => {
        console.log("key", `${a}`)
        tempColorArray[i] = palette[`${a}`]
      })
    }
    if (type === 'continuous') {
      //TODO: get min/max, convert to percentages
      startArray.forEach((a, i) => {
        let rgb = blendColors(hexToRgb(palette['0%']), hexToRgb(palette['100%']), a)
        tempColorArray[i] = 'rgb(' + rgb.r + ', ' + rgb.g + ', ' + rgb.b + ')';
      })
    }
    console.log("color array", tempColorArray)
    setColorArray([...tempColorArray])
  }

  const jumpToNextInstance = (forward) => {
    console.log("jump func", forward)
    const condition = (e) => {
      console.log("result", e, type === 'continuous'
      ? e < 0.8
      : e === 'away')
      return type === 'continuous' 
        ? e < 0.8
        : e === 'away'
    } 
    let next = -1
     if(forward === true) {
      let arraySlice = videoData.annotations[id].slice(playbackState.currentFrame + 1)
      next = arraySlice.findIndex((e) => condition(e))
      if (next !== -1) { next = next + playbackState.currentFrame + 1}
      
    } else {
      let arraySlice = videoData.annotations[id].slice(0, playbackState.currentFrame)
      next = arraySlice.findLastIndex((e) => condition(e))
     }
     if (next !== -1) {
      dispatchPlaybackState({
        type: 'setCurrentFrame',
        currentFrame: next
       })
       console.log("jumping to ", next)
     } 
  }
  
  return (
    <div>
      <HeatmapCanvas colorArray={colorArray}/>
      <JumpButton handleJump={jumpToNextInstance} />
    </div>
  );
}
  
export default AnnotationBar;

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