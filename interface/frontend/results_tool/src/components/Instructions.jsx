import { 
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Typography
} from '@mui/material';
import { 
  ExpandMore
} from '@mui/icons-material';

import styles from './Instructions.module.css';

function Instructions() {

  const styleOverrides = {
    accordion: {
      backgroundColor: 'lightgray'
    },
    accordionSummary : {
      alignItems: 'center',
      justifyContent: 'flex-start',
      margin: '12px 0',
      minHeight: 0,
      '.MuiAccordionSummary-content': {
        display: 'flex',
        flexFlow: 'column nowrap',
        justifyContent: 'center',
        alignItems: 'end',
        margin: 0
      }
    },
    accordionDetails: {
      backgroundColor: '#eee',
      color: 'black',
      padding: 0,
      borderTop: '1px solid darkgray'
    }
  }

  return (
    <Accordion 
      sx={styleOverrides.accordion}
      disableGutters
      square
    >
        <AccordionSummary
          expandIcon={<ExpandMore />}
          sx={styleOverrides.accordionSummary}
        >
            <Typography variant="button" className={styles.buttonText}>Instructions</Typography>
        </AccordionSummary>
        <AccordionDetails
          sx={styleOverrides.accordionDetails}
        >
          <div className={styles.instructionContainer}>
            {/* <h4 className={styles.instructionsHeader}>Welcome to iCatcher+</h4> */}
            <p className={styles.instructions}>Click the video or press SPACE to start play</p>
            <p className={styles.instructions}>Use the 'r' key to play the video in reverse</p>
            <p className={styles.instructions}>Use the left and right arrow keys to page through the video frames</p>
            <p className={styles.instructions}>Jump to a specific frame using the input box above the video</p>
          </div>
        </AccordionDetails>
      </Accordion>
  );
}

export default Instructions;
