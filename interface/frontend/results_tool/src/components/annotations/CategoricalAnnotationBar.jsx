import {
  MenuItem,
  TextField
} from '@mui/material'
import { useEffect, useState } from 'react';
import { addSnack, useSnacksDispatch } from '../../state/SnacksProvider';
import { useVideoData, useVideoDataDispatch } from '../../state/VideoDataProvider';
import { usePlaybackState, usePlaybackStateDispatch } from '../../state/PlaybackStateProvider';
import HeatmapCanvas from './HeatmapCanvas';
import JumpButton from '../common/JumpButton';
import AnnotationBar from './AnnotationsBar';


import styles from './ContinuousAnnotationBar.module.css'


/* Expected props:
tbd
*/

const colorPalettes = {
  machineLabel: {
    'undefined': '#C5C5C5',
    'left': '#F05039',
    'right': '#1F449C',
    'away': '#EEBAB4',
    'noface': '#7CA1CC',
    'nobabyface': '#000000'
  },
  edited: {
    'edited': '#F05039',
    'unedited':'#FFFFFF'
  },
  default: {
    '0': '#C5C5C5',
    '1': '#F05039',
    '2': '#1F449C',
    '3': '#EEBAB4',
    '4': '#7CA1CC',
    '5': '#000000'
  }
}

function CategoricalAnnotationBar(props) {
  
  const { id } = props;
  const videoData = useVideoData();
  const playbackState = usePlaybackState();
  const dispatchPlaybackState = usePlaybackStateDispatch();

  const [ colorPalette, setColorPalette ] = useState(id);
  const [ selectedLabel, setSelectedLabel ] = useState('away'); //fix
  const [ labelOptions, setLabelOptions ] = useState([])

  useEffect (()=> {
    let labels = [ ...new Set(videoData.annotations[id].slice(videoData.metadata.frameOffset))].map((l) => {return String(l)})
    setLabelOptions(labels)
    setSelectedLabel(labels[0])
    setColorPalette(id)
    console.log("labelOptions", labels)
    
  }, [id, videoData.annotations])

  const computeColorArray = () => {
    console.log('running cat computeColorArray')
    let tempColorArray = []
    const palette = colorPalettes[colorPalette || 'default']
    const annotationArray = videoData.annotations[id].slice(videoData.metadata.frameOffset)
    annotationArray.forEach((a, i) => {
    // videoData.annotations[id].forEach((a, i) => {
      // console.log("key", `${a}`)
      tempColorArray[i] = palette[`${a}`]
    })
    // console.log("color array", tempColorArray)
    return [...tempColorArray]
  }

  const jumpToNextInstance = (forward) => {
    // console.log("jump func", forward)
    const condition = (e) => {
      return selectedLabel === 'undefined'
        ? e === undefined
        : e === selectedLabel
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
      //  console.log("jumping to ", next)
     } 
  }

  
  const handleLabelChange = (e) => {
    let targetValue = e.target.value
    if(!(targetValue in labelOptions)) {
      addSnack(`Value is not a valid ${id} option`, 'warning')
      return
    }
    setSelectedLabel(targetValue)
  }
  
  const getThresholdInput = () => {
    return <TextField
            className={styles.threshold}
            id={`${id}-threshold-jumper`}
            label={"Label"}
            select
            margin="dense"
            multiline={false}
            defaultValue={selectedLabel}
            onChange={handleLabelChange}
          >
            {labelOptions.map((option) => (
                <MenuItem key={option} value={option}>
                  {option}
                </MenuItem>
              ))}
          </TextField>
  }

  return (
    <div className={styles.temp}>
      <AnnotationBar 
        id={id}
        getColorArray={computeColorArray}
        handleJump={jumpToNextInstance}
        getThresholdInput={getThresholdInput}
      />
    </div>
  );
}
  
export default CategoricalAnnotationBar;
