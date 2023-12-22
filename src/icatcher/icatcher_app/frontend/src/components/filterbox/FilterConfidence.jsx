import {
  TextField
} from '@mui/material';
import { useEffect, useState } from 'react';
import { addSnack, useSnacksDispatch } from '../../state/SnacksProvider';
import { useVideoData } from '../../state/VideoDataProvider';
import { usePlaybackState, usePlaybackStateDispatch } from '../../state/PlaybackStateProvider';
import SingleFilter from './SingleFilter';

/* Expected props:
  id:string --> key for annotation array in videoData.annotations
*/
function FilterConfidence(props) {
  
  const { id } = props;
  const videoData = useVideoData();
  const playbackState = usePlaybackState();
  const dispatchPlaybackState = usePlaybackStateDispatch();
  const dispatchSnack = useSnacksDispatch();

  const [ threshold, setThreshold ] = useState(0.80);
  const [ range, setRange ] = useState({ min: 0, max: 1})

  useEffect (()=> {
    let [ newMin, newMax ] = getRange()
    setRange({min: newMin, max: newMax})
    if(newMin > threshold || newMax < threshold) {
      setThreshold((newMax-newMin)/2)
    }
  }, [id, videoData.annotations])  // eslint-disable-line react-hooks/exhaustive-deps

  const jumpToNextInstance = (forward) => {
    const condition = (e) => {
      return e < threshold
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
     } 
  }

  const handleThresholdChange = (e) => {
    let targetValue = e.target.value
    if(targetValue < range.min) {
      addSnack(`Value is below minimum ${id}. Defaulting to minimum ${range.min}`, 'info')
      targetValue = range.min
    }
    else if (targetValue > range.max) {
      dispatchSnack(addSnack(`Value is above maximum ${id}. Defaulting to maximum ${range.max}`, 'info'))
      targetValue = range.max
    }
    setThreshold(targetValue)
  }
  
  return (
    <div>
      <SingleFilter
        handleJump={jumpToNextInstance}
      >
        <TextField
          id={`${id}-threshold-jumper`}
          label={"Threshold"}
          name="Name"
          type="number"
          inputProps={{
            min: range[0],
            max: range[1],
            step: 0.01,
          }}
          margin="dense"
          multiline={false}
          value={threshold}
          onChange={handleThresholdChange}
          style={{width:80}}
        />
      </SingleFilter>
    </div>
  );
}
  
export default FilterConfidence;

const getRange = () => {
  const min = 0
  const max = 1
  return [min, max]
}