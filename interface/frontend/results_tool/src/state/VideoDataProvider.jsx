import { createContext, useContext, useReducer } from 'react';

const VideoDataContext = createContext(null);

const VideoDataDispatchContext = createContext(null);

export function VideoDataProvider({ children }) {
  const [videoData, dispatch] = useReducer(
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
      console.log("State update for frames", [...action.frames])
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
      return { 
        metadata: {},
        frames: [],
        annotations: []
      };
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

/* Example metadata:
    {   ...
        "baseFramePath": "./ic_test_output_jpeg",
        "baseFileName": "frm",
        "numDigitsFrame": 3,
        "frameExt": ".jpeg"
    }
    would point to frames starting at './ic_test_output_jpeg/frm_001.jpeg'
 */

