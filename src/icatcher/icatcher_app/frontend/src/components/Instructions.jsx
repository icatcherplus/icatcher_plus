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
      backgroundColor: 'lightblue'
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
            <p className={styles.instructions}>Click the video or press SPACE to start play</p>
            <p className={styles.instructions}>Jump to a specific frame using the input box above the video</p>
            <p className={styles.instructions}>To page through the video frames, use the left and right buttons on the video control bar or your left and right arrow keys.</p>
            <p className={styles.instructions}>Use the 's' key or the circular button on the video control bar to play the video in slow motion</p>
          </div>
        </AccordionDetails>
      </Accordion>
  );
}

export default Instructions;
