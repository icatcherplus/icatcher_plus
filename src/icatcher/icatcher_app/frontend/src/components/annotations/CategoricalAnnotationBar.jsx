import {
  MenuItem,
  Select
} from '@mui/material'
import { useEffect, useState } from 'react';
import { addSnack, useSnacksDispatch } from '../../state/SnacksProvider';
import { useVideoData } from '../../state/VideoDataProvider';
import { usePlaybackState, usePlaybackStateDispatch } from '../../state/PlaybackStateProvider';
import AnnotationBar from './AnnotationsBar';
import styles from './ContinuousAnnotationBar.module.css'

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
  const dispatchSnack = useSnacksDispatch();

  const [ colorPalette, setColorPalette ] = useState(id);
  const [ selectedLabel, setSelectedLabel ] = useState('away');
  const [ labelOptions, setLabelOptions ] = useState([])

  useEffect (()=> {
    let labels = [ ...new Set(videoData.annotations[id].slice(videoData.metadata.frameOffset))].map((l) => {return String(l)})
    setLabelOptions(labels)
    setSelectedLabel(labels[0])
    setColorPalette(id)
    
  }, [id, videoData.annotations])  // eslint-disable-line react-hooks/exhaustive-deps

  const computeColorArray = () => {
    let tempColorArray = []
    const palette = colorPalettes[colorPalette || 'default']
    const annotationArray = videoData.annotations[id].slice(videoData.metadata.frameOffset)
    annotationArray.forEach((a, i) => {
      tempColorArray[i] = palette[`${a}`]
    })
    return [...tempColorArray]
  }

    const jumpToNextInstance = (forward) => {
    const condition = (e) => { 
      return selectedLabel === 'undefined'
        ? e === undefined
        : e === selectedLabel
    } 
    let next = -1
     if(forward === true) {
      let arraySlice = videoData.annotations[id].slice(playbackState.currentFrame + 1)

      const firstFrame = 7; //DO NOT KEEP AS 7. ONLY FOR THIS IMPLEMENTATION.
      // console.log(playbackState.currentFrame)
      if (playbackState.currentFrame > firstFrame) { 
        if (!condition(videoData.annotations[id][playbackState.currentFrame-1]) && !condition(videoData.annotations[id][playbackState.currentFrame+1])) { 
          console.log("single frame");
          next = arraySlice.findIndex((e) => condition(e))
        } else if (!condition(videoData.annotations[id][playbackState.currentFrame-1])) { //Indicates a start. Find the next frame that isn't the same, and then -1 to get the last one that is.
          // console.log("start")
          next = arraySlice.findIndex((e) => !condition(e)) - 1
        } else {
          // console.log("end")
          next = arraySlice.findIndex((e) => condition(e))
        }
      } else {
        // console.log("first frame");
        if (condition(videoData.annotations[id][playbackState.currentFrame])) { //First frame fits condition. Find the end.
          next = arraySlice.findIndex((e) => !condition(e)) - 1
        } else {
          next = arraySlice.findIndex((e) => condition(e))
        }
      }
      if (next !== -1) { next = next + playbackState.currentFrame + 1}
       
    } else {
      let arraySlice = videoData.annotations[id].slice(0, playbackState.currentFrame)

      const lastFrame = videoData.annotations[id].length - 8; //Subtracting by 7 (the offset) and 1 (1-indexing instead of 0-indexing)
      if (playbackState.currentFrame < lastFrame) {
        if (!condition(videoData.annotations[id][playbackState.currentFrame-1]) && !condition(videoData.annotations[id][playbackState.currentFrame+1])) { 
          // console.log("single frame");
          next = arraySlice.findLastIndex((e) => condition(e))
        } else if (!condition(videoData.annotations[id][playbackState.currentFrame+1])) { //Indicates a start. Find the next frame that isn't the same, and then -1 to get the last one that is.
          // console.log("start")
          next = arraySlice.findLastIndex((e) => !condition(e)) + 1
        } else {
          // console.log("end")
          next = arraySlice.findLastIndex((e) => condition(e))
        }
      } else {
        // console.log("last frame");
        if (condition(videoData.annotations[id][playbackState.currentFrame])) { //Last frame fits condition. Find the end of the block.
          next = arraySlice.findLastIndex((e) => !condition(e)) + 1
        } else {
          next = arraySlice.findLastIndex((e) => condition(e))
        }
      }
     }
     if (next !== -1) {
      dispatchPlaybackState({
        type: 'setCurrentFrame',
        currentFrame: next
       })
     } 
  }

  
  const handleLabelChange = (e) => {
    let targetValue = e.target.value
    if(!(labelOptions.includes(targetValue))) {
      dispatchSnack(addSnack(`Value is not a valid ${id} option`, 'warning'))
      return
    }
    setSelectedLabel(targetValue)
    
  }

  return (
    <div className={styles.temp}>
      <AnnotationBar 
        id={id}
        getColorArray={computeColorArray}
        handleJump={jumpToNextInstance}
      >
        { labelOptions.length === 0
          ? <div />
          : <Select
            className={styles.threshold}
            id={`${id}-threshold-jumper`}
            label={"Label"}
            margin="dense"
            defaultValue={selectedLabel}
            value={selectedLabel}
            onChange={(e) =>handleLabelChange(e)}
          >
            {labelOptions.map((option) => (
                <MenuItem key={option} value={option}>
                  {option}
                </MenuItem>
              ))}
          </Select>
        }
        
      </AnnotationBar>
    </div>
  );
}
  
export default CategoricalAnnotationBar;