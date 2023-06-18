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
          metadata: { ...action.metadata}
        };
    }
    default: {
      throw Error('Unknown action: ' + action.type);
    }
  }
}
const initialPlaybackState = {
  currentFrame: undefined,
  forwardPlay: true,
  slowMotion: false,
  // playTimer: undefined
}
