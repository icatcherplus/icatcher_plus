import { createContext, useContext } from 'react';
import { useReducerWithThunk } from './utils/useReducerWithThunk'

const VideoDataContext = createContext(null);

const VideoDataDispatchContext = createContext(null);

export function VideoDataProvider({ children }) {
  const [videoData, dispatch] = useReducerWithThunk(
    videoDataReducer,
    initialVideoData
  );

  return (
    <VideoDataContext.Provider value={videoData}>
      <VideoDataDispatchContext.Provider value={dispatch}>
        {children}
      </VideoDataDispatchContext.Provider>
    </VideoDataContext.Provider>
  );
}

export function useVideoData() {
  return useContext(VideoDataContext);
}

export function useVideoDataDispatch() {
  return useContext(VideoDataDispatchContext);
}

function videoDataReducer(videoData, action) {
  switch (action.type) {
    case 'setMetadata': {
      return { ...videoData,
          metadata: { ...action.metadata}
        };
    }
    case 'setFrames': {
      return { ...videoData,
        frames: [...action.frames]
      }
    }
    case 'setAnnotations': {
      return { ...videoData,
        annotations: { ...action.annotations}
      }
    }
    case 'resetVideo': {
      return initialVideoData;
    }
    default: {
      throw Error('Unknown action: ' + action.type);
    }
  }
}

const initialVideoData = {
  metadata: {},
  frames: [],
  annotations: []
}

export const METADATA_FIELD_MAPPING = {
  framesPerSecond: "fps",
  numFrames: "numFrames",
  frameOffset: "frameOffset"
}
