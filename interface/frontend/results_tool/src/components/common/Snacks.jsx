import { 
  Alert,
  Grow,
  Snackbar
} from '@mui/material';
import { useState, useEffect } from 'react';
import { useSnacks, useSnackDispatch } from '../../state/SnackContext';

/* Expected props:
    none
 */
function Snacks() {

  const snacks = useSnacks();
  const dispatch = useSnackDispatch();
  const [ open, setOpen ] = useState()

  useEffect (() => {
    setOpen(snacks.length !== 0)
  }, [ snacks ])

  const handleClose = (e, reason) => {
    if (reason === 'clickaway') {
      return;
    }
    dispatch({
      type: "removeTopSnack"    
    });
  };

  return (
    <Snackbar
      anchorOrigin={{ vertical:'top', horizontal:'center' }}
      open={open}
      autoHideDuration={3000}
      onClose={handleClose}
      TransitionComponent={Grow}
    >
      <Alert severity={snacks[0]?.severity}>
        {snacks[0]?.message}
      </Alert>
    </Snackbar>
);
}

export default Snacks;
