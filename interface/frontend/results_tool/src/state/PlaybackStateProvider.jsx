import { createContext, useContext } from 'react';
import { useReducerWithThunk } from './utils/useReducerWithThunk'

const PlaybackStateContext = createContext(null);

const PlaybackStateDispatchContext = createContext(null);

export function PlaybackStateProvider({ children }) {
  const [playbackState, dispatch] = useReducerWithThunk(
    playbackStateReducer,
    initialPlaybackState
  );

  return (
    <PlaybackStateContext.Provider value={playbackState}>
      <PlaybackStateDispatchContext.Provider value={dispatch}>
        {children}
      </PlaybackStateDispatchContext.Provider>
    </PlaybackStateContext.Provider>
  );
}

export function usePlaybackState() {
  return useContext(PlaybackStateContext);
}

export function usePlaybackStateDispatch() {
  return useContext(PlaybackStateDispatchContext);
}

function playbackStateReducer(playbackState, action) {
  switch (action.type) {
    case 'setCurrentFrame': {
      return { ...playbackState,
        currentFrame: action.currentFrame
      };
    }
    case 'setAspectRatio': {
      return { ...playbackState,
        aspectRatio: action.aspectRatio
      };
    }
    case 'setVideoWidth': {
      console.log('videoWidth', action.videoWidth)
      return { ...playbackState,
        width: action.videoWidth
      };
    }
    case 'setForwardPlay': {
      return { ...playbackState,
        forwardPlay: action.forwardPlay
      };
    }
    case 'setSlowMotion': {
      return { ...playbackState,
        slowMotion: action.slowMotion
      };
    }
    case 'setPaused': {
      return { ...playbackState,
        paused: action.paused
      };
    }
    case 'setFrameRange': {
      return { ...playbackState,
        frameRange: action.frameRange
      };
    }
    default: {
      throw Error('Unknown action: ' + action.type);
    }
  }
}
const initialPlaybackState = {
  currentFrame: undefined,
  aspectRatio: 16/9,
  videoWidth: window.innerHeight * .8,
  forwardPlay: true,
  slowMotion: false,
  paused: true,
  frameRange: [null, null]
}


export const updateVideoWidth = () => (dispatch, getState) => {
  let state = getState()
  let videoWidth = (window.innerHeight * 0.8) * state.aspectRatio
  if (videoWidth > (0.8 * window.innerWidth)) {
    videoWidth = (0.8 * window.innerWidth)
  }
  dispatch({
    type: 'setVideoWidth',
    videoWidth: videoWidth
  })
}

export const updateDimensions = (aspectRatio) => (dispatch, getState) => {
  console.log("updating dimensions")
  dispatch({
    type: 'setAspectRatio',
    aspectRatio: aspectRatio
  })
  dispatch(updateVideoWidth());
}

export const getNextFrame = (frameDeck) => (dispatch, getState) => {
  let state = getState()
  let nextFrame = state.forwardPlay 
    ? state.currentFrame + 1 
    : state.currentFrame - 1
  if (
    typeof(frameDeck[nextFrame]) === 'undefined' 
    || frameDeck[nextFrame].loaded === false
  ) {
    dispatch({
      type: 'setPaused',
      paused: true
    })
  } else {
    dispatch({
      type: 'setCurrentFrame',
      currentFrame: nextFrame
    })
  }
}

export const updateFrameRange = (targetFrameRange, videoData, snackCallback) => (dispatch) => {
  let [ minFrame, maxFrame ] = getMinMaxFrames(videoData)
  console.log("here", minFrame, maxFrame)
  let tempFrameRange = targetFrameRange.map(frameNumber => {
    if(frameNumber == null) {
      return minFrame
    }
    if (frameNumber < minFrame) {
      snackCallback(`Cannot scrub to invalid frame ${frameNumber}`, 'warning')
      return minFrame
    } 
    if (frameNumber > maxFrame) {
      snackCallback(`Cannot scrub to invalid frame ${frameNumber}`, 'warning')
      return maxFrame
    }
    return frameNumber
  })
  dispatch({
    type: 'setFrameRange',
    frameRange: tempFrameRange
  })
}

const getMinMaxFrames = (videoData) => {
  return [
    videoData.metadata.frameOffset,
    videoData.metadata.numFrames - 1
  ]
}