import { createContext, useContext, useReducer } from 'react';

const PlaybackStateContext = createContext(null);

const PlaybackStateDispatchContext = createContext(null);

export function PlaybackStateProvider({ children }) {
  const [playbackState, dispatch] = useReducer(
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
        currentFrame: { ...action.currentFrame}
        };
    }
    case 'setAspectRatio': {
      return { ...playbackState,
        aspectRatio: { ...action.aspectRatio}
        };
    }
    case 'setWidth': {
      return { ...playbackState,
        width: { ...action.width}
        };
    }
    case 'setForwardPlay': {
      return { ...playbackState,
        forwardPlay: { ...action.forwardPlay}
        };
    }
    case 'setSlowMotion': {
      return { ...playbackState,
        slowMotion: { ...action.slowMotion}
        };
    }
    case 'setPaused': {
      return { ...playbackState,
        paused: { ...action.paused}
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
  width: undefined,
  forwardPlay: true,
  slowMotion: false,
  paused: true
  // playTimer: undefined
}
