import { useReducer, useRef } from 'react'

export function useReducerWithThunk(reducer, initialState) {
  const [state, dispatch] = useReducer(reducer, initialState);

  const stateRef = useRef();

  const customDispatch = (action) => {

    if (typeof action === 'function') {
      action(customDispatch, () => stateRef.current);
    } else {
      dispatch(action); 
    }
  };

  stateRef.current = state
  return [state, customDispatch];
}
