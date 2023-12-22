import React from 'react';
import SingleFilter from './SingleFilter'
import {
  MenuItem,
  Select
} from '@mui/material'
import { useEffect, useState } from 'react';
import { addSnack, useSnacksDispatch } from '../../state/SnacksProvider';
import { useVideoData } from '../../state/VideoDataProvider';
import { usePlaybackState, usePlaybackStateDispatch } from '../../state/PlaybackStateProvider';

function FilterGaze(props) {

  const { id } = props;
  const videoData = useVideoData();
  const playbackState = usePlaybackState();
  const dispatchPlaybackState = usePlaybackStateDispatch();
  const dispatchSnack = useSnacksDispatch();

  const [ selectedLabel, setSelectedLabel ] = useState('away');
  const [ labelOptions, setLabelOptions ] = useState([])

  useEffect (()=> {
    let labels = [ ...new Set(videoData.annotations[id].slice(videoData.metadata.frameOffset))].map((l) => {return String(l)})
    setLabelOptions(labels)
    setSelectedLabel(labels[0])

  }, [id, videoData.annotations])  // eslint-disable-line react-hooks/exhaustive-deps

  const jumpToNextInstance = (forward) => {
    const condition = (e) => { 
      return selectedLabel === 'undefined'
        ? e === undefined
        : e === selectedLabel
    } 
    let next = -1
     if(forward === true) {
      let arraySlice = videoData.annotations[id].slice(playbackState.currentFrame + 1)

      if (!condition(videoData.annotations[id][playbackState.currentFrame])) {
        next = arraySlice.findIndex((e) => condition(e))
      } else if (playbackState.currentFrame > 0) { 
        if (!condition(videoData.annotations[id][playbackState.currentFrame-1]) && !condition(videoData.annotations[id][playbackState.currentFrame+1])) { 
          next = arraySlice.findIndex((e) => condition(e))
        } else if (!condition(videoData.annotations[id][playbackState.currentFrame-1]) || (condition(videoData.annotations[id][playbackState.currentFrame-1]) && condition(videoData.annotations[id][playbackState.currentFrame+1]))) { 
          if (arraySlice.findIndex((e) => !condition(e)) === -1) {
            next = arraySlice.length - 1
          } else {
            next = arraySlice.findIndex((e) => !condition(e)) - 1
          }
        } else {
          next = arraySlice.findIndex((e) => condition(e))
        }
      } else {
        if (condition(videoData.annotations[id][playbackState.currentFrame])) { 
          next = arraySlice.findIndex((e) => !condition(e)) - 1
        } else {
          next = arraySlice.findIndex((e) => condition(e))
        }
      }
      if (next !== -1) { next = next + playbackState.currentFrame + 1}
       
    } else {
      let arraySlice = videoData.annotations[id].slice(0, playbackState.currentFrame)

      const lastFrame = videoData.annotations[id].length - 1;
      if (!condition(videoData.annotations[id][playbackState.currentFrame])) {
        next = arraySlice.findLastIndex((e) => condition(e))
      } else if (playbackState.currentFrame < lastFrame) {
        if (!condition(videoData.annotations[id][playbackState.currentFrame-1]) && !condition(videoData.annotations[id][playbackState.currentFrame+1])) { 
          next = arraySlice.findLastIndex((e) => condition(e))
        } else if (!condition(videoData.annotations[id][playbackState.currentFrame+1]) || (condition(videoData.annotations[id][playbackState.currentFrame-1]) && condition(videoData.annotations[id][playbackState.currentFrame+1]))) { 
          next = arraySlice.findLastIndex((e) => !condition(e)) + 1
        } else {
          next = arraySlice.findLastIndex((e) => condition(e))
        }
      } else {
        if (condition(videoData.annotations[id][playbackState.currentFrame])) {
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
    <div>
      <SingleFilter
        handleJump={jumpToNextInstance}
      >
        { labelOptions.length === 0
          ? <div />
          : <Select
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

      </SingleFilter>
    </div>
  );
}

export default FilterGaze