import { 
    Dialog,
    DialogTitle,
    DialogContent
  } from '@mui/material';
  import { useRef } from 'react';
  import { useSnackDispatch } from '../../state/SnackContext';
  
  /* Expected props:
    tbd
  */
  function KeystrokeMonitor(props, {children}) {
  
    const { tbd } = props;
    const dispatchSnack = useSnackDispatch();

    let shiftPressed = false;
  
  
    return (
      <KeystrokeMonitor>
        {children}
      </KeystrokeMonitor>
    );
  }
  
  export default KeystrokeMonitor;