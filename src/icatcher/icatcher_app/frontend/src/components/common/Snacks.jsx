import { 
  Alert,
  Slide,
  Snackbar
} from '@mui/material';
import { useSnacks } from '../../state/SnacksProvider';

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
    >
      <Alert variant="filled" severity={snacks[0]?.severity}>
        {snacks[0]?.message}
      </Alert>
    </Snackbar>
);
}

export default Snacks;