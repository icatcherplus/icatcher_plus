import { createContext, useContext } from 'react';
import { useReducerWithThunk } from './utils/useReducerWithThunk'

const ViewContext = createContext(null);

const ViewDispatchContext = createContext(null);

export function ViewProvider({ children }) {
  const [view, dispatch] = useReducerWithThunk(
    viewReducer,
    initialView
  );

  return (
    <ViewContext.Provider value={view}>
      <ViewDispatchContext.Provider value={dispatch}>
        {children}
      </ViewDispatchContext.Provider>
    </ViewContext.Provider>
  );
}

export function useView() {
  return useContext(ViewContext);
}

export function useViewDispatch() {
  return useContext(ViewDispatchContext);
}

function viewReducer(view, action) {
  switch (action.type) {
    case 'setView': {
     return {
      ...view,
      currentView: action.currentView
     }
    }
    default: {
      throw Error('Unknown action: ' + action.type);
    }
  }
}

export const VIEWS = [
  'UPLOAD',
  'PROGRESS',
  'RESULTS'
]

const initialView = {
  currentView: VIEWS[2]
}

export const updateView = (view) => dispatch => {
  if (!VIEWS.includes(view)) {
    console.error(`${view} is not a valid view name`)
  } else {
    dispatch({
      type: 'setView',
      currentView: view
    })
  }
}