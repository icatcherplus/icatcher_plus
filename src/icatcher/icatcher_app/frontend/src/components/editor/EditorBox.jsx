import React from 'react';
import styles from './EditorBox.module.css';
import { Button, Select, MenuItem, TextField } from '@mui/material';

const labelOptions = ['none', 'left', 'right', 'away', 'noface'];
const includedOptions = ['yes', 'no'];

const styleOverrides = {
  commentField: {
    width: '100%',
    // textfield padding
    '& .css-dpjnhs-MuiInputBase-root-MuiOutlinedInput-root': {
      padding: '7px',
    },
  },

  frameInputField: {
  // textfield padding
  '& .css-1t8l2tu-MuiInputBase-input-MuiOutlinedInput-input': {
    padding: '5px',
    }
  },

  // change comment input styling
  commentText: {
    style: {
      color: 'gray',
      fontSize: 15,
      textAlign: 'left',
    },
  },

  frameInputText: {
    style: {
      color: 'gray',
      fontSize: 15,
      textAlign: 'center',
    },
  },
}

function EditorBox() {
  return (
  <React.Fragment>

    <div className={styles.box}>

      <div className={styles.frame}>
        <p className={styles.frameLabel}>Frame: </p>
        <div className={styles.frameSelect}>
          <TextField
            inputProps={styleOverrides.frameInputText}
            sx={styleOverrides.frameInputField}
            id='frame-start-range'
            variant='outlined'/>

          <p className={styles.marginLeftRight}>
          to
          </p>

          <TextField
            inputProps={styleOverrides.frameInputText}
            sx={styleOverrides.frameInputField}
            id='frame-end-range'
            variant='outlined'/>
        </div>
      </div>

      <div className={styles.label}>
        Label:
        <Select
          className={styles.select}
          id='editor-select-label'
          >
            {labelOptions.map((option) => (<MenuItem value={option}>{option}</MenuItem>))}
        </Select>
      </div>

      <div className={styles.included}>
        Included:
        <Select
          className={styles.select}
          id='editor-select-included'>
            {includedOptions.map((option) => (<MenuItem value={option}>{option}</MenuItem>))}
        </Select>
      </div>

      <div className={styles.comments}>
        Comments:
        <TextField
          inputProps={styleOverrides.commentText}
          sx={styleOverrides.commentField}
          margin='dense'
          id='comments'
          variant='outlined'
          multiline
          maxRows={4} />
      </div>

      <Button variant='outlined' size='small'>Submit</Button>

    </div>

  </React.Fragment>
  )
}

export default EditorBox