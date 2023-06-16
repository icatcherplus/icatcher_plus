import { createContext, useContext, useReducer } from 'react';

const SnackContext = createContext(null);

const SnackDispatchContext = createContext(null);

export function SnackProvider({ children }) {
  const [snacks, dispatch] = useReducer(
    snackReducer,
    initialSnacks
  );

  return (
    <SnackContext.Provider value={snacks}>
      <SnackDispatchContext.Provider value={dispatch}>
        {children}
      </SnackDispatchContext.Provider>
    </SnackContext.Provider>
  );
}

export function useSnacks() {
  return useContext(SnackContext);
}

export function useSnackDispatch() {
  return useContext(SnackDispatchContext);
}

const SNACK_SEVERITY = [
  'error',
  'info',
  'success',
  'warning'
]

function snackReducer(snacks, action) {
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