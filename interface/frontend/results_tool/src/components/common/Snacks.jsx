import { 
  Alert,
  Slide,
  Snackbar
} from '@mui/material';
import { useState, useEffect } from 'react';
import { useSnacks, useSnacksDispatch } from '../../state/SnacksProvider';

/* Expected props:
    none
 */
function Snacks() {

  const snacks = useSnacks();

  return (
    <Snackbar
      anchorOrigin={{ vertical:'top', horizontal:'center' }}
      open={snacks.length !== 0}
      TransitionComponent={Slide}
      // message={snacks[0]?.message}
    >
      <Alert variant="filled" severity={snacks[0]?.severity}>
        {snacks[0]?.message}
      </Alert>
    </Snackbar>
);
}

export default Snacks;