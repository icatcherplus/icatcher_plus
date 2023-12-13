import React, { useRef } from 'react';
import {
  Skeleton
} from '@mui/material';
import { useVideoData } from '../../state/VideoDataProvider';
import { usePlaybackStateDispatch } from '../../state/PlaybackStateProvider';
import FrameInfo from './FrameInfo';


function FrameInfoFrame() {
  const videoData = useVideoData();

  const frameImages = useRef([]);
  const dispatchPlaybackState = usePlaybackStateDispatch();
  const pause = () => {
    dispatchPlaybackState({
      type: 'setPaused',
      paused: true
    })
  }

  const showFrame = (index) => {
    pause();
    if ((typeof (frameImages.current[index]) === 'undefined') || (frameImages.current[index].loaded === false)) {
      return;
    }
    dispatchPlaybackState({
      type: 'setCurrentFrame',
      currentFrame: index
    })
  }

  return (
    <React.Fragment>
    <div>
    {
      Object.keys(videoData.annotations).length !== 0 ?
      <div>
        <FrameInfo
          handleJumpToFrame={(i) => showFrame( (i))}
        />
      </div>
      :
      <Skeleton
            variant="text"
            height={200}
          />
    }
    </div>
    </React.Fragment>
  );
}

export default FrameInfoFrame;