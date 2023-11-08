import { createContext, useContext } from 'react';
import { useReducerWithThunk } from './utils/useReducerWithThunk'


const SnacksContext = createContext(null);

const SnacksDispatchContext = createContext(null);

export function SnacksProvider({ children }) {
  const [snacks, dispatch] = useReducerWithThunk(
    snacksReducer,
    initialSnacks
  );

  return (
    <SnacksContext.Provider value={snacks}>
      <SnacksDispatchContext.Provider value={dispatch}>
        {children}
      </SnacksDispatchContext.Provider>
    </SnacksContext.Provider>
  );
}

export function useSnacks() {
  return useContext(SnacksContext);
}

export function useSnacksDispatch() {
  return useContext(SnacksDispatchContext);
}

const SNACK_SEVERITY = [
  'error',
  'info',
  'success',
  'warning'
]

function snacksReducer(snacks, action) {
  switch (action.type) {
    case 'pushSnack': {
      let severity = action.severity
      if (!SNACK_SEVERITY.includes(severity)) {
        severity = 'info'
        console.warn(`${action.severity} is not a valid severity setting for snack "${action.message}"`)
      }
      return [...snacks, 
        { severity: severity, 
          message: action.message
        }]
    }
    case 'removeTopSnack': {
      return snacks.slice(1);
    }
    default: {
      throw Error('Unknown action: ' + action.type);
    }
  }
}

const initialSnacks = []

export const addSnack = (message, severity) => dispatch => {
  dispatch({
    type: 'pushSnack',
    severity: severity,
    message: message
  })
  setTimeout(() => {
    dispatch({
      type: 'removeTopSnack'
    })
  }, 1000)
}