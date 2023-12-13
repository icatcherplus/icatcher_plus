import React, { useState } from 'react';
import styles from './EditorBox.module.css';
import { Button, Select, MenuItem, TextField, Skeleton } from '@mui/material';
import { useVideoData } from '../../state/VideoDataProvider';

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
  submitButton: {
    backgroundColor: "lightblue",
    color: "gray",
    borderColor: "lightblue",
  },
}

function EditorBox() {

  const videoData = useVideoData();
  const [selectedLabel, setSelectedLabel] = useState('');
  const [selectedInclude, setSelectedInclude] = useState('');

  const handleSelectLabelChange = (event) => {
    setSelectedLabel(event.target.value);
  };
  const handleSelectIncludeChange = (event) => {
    setSelectedInclude(event.target.value);
  };

  return (
  <React.Fragment>
  { Object.keys(videoData.annotations).length !== 0 ?
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
          value={''}
          onChange={handleSelectLabelChange}
          >
          {labelOptions.map((option) => (
            <MenuItem key={option} value={option}>
              {option}
            </MenuItem>
          ))}
        </Select>
      </div>

      <div className={styles.included}>
        Included:
        <Select
          className={styles.select}
          id='editor-select-included'
          value={''}
          onChange={handleSelectIncludeChange}
          >
          {includedOptions.map((option) => (
            <MenuItem key={option} value={option}>
              {option}
            </MenuItem>
          ))}
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

      <Button
        variant='outlined'
        size='medium'
        sx={styleOverrides.submitButton}
      >
        Submit
      </Button>

    </div>

    :
    <Skeleton
      variant="text"
//       width={}
      height={200}
    />
  }
  </React.Fragment>
  )
}

export default EditorBox