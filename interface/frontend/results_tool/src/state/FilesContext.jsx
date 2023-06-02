import { createContext, useContext, useReducer } from 'react';

const FilesContext = createContext(null);

const FilesDispatchContext = createContext(null);

export function FilesProvider({ children }) {
  const [files, dispatch] = useReducer(
    filesReducer,
    initialFiles
  );

  return (
    <FilesContext.Provider value={files}>
      <FilesDispatchContext.Provider value={dispatch}>
        {children}
      </FilesDispatchContext.Provider>
    </FilesContext.Provider>
  );
}

export function useFiles() {
  return useContext(FilesContext);
}

export function useFilesDispatch() {
  return useContext(FilesDispatchContext);
}

function filesReducer(files, action) {
  switch (action.type) {
    case 'addMetadata': {
      return {...files, metadata: action.file };
    }
    case 'addVideo': {
        return {...files, video: action.file };
    }
    case 'addFrames': {
        return {...files, frames: action.files };
    }
    case 'addAnnotations': {
        return {...files, frames: action.file };
    }
    default: {
      throw Error('Unknown action: ' + action.type);
    }
  }
}

const initialFiles = {
    metadata: undefined,
    video: undefined,
    frames: undefined,
    annotations: undefined
}