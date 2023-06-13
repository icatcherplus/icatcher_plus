import { 
  Dialog,
  DialogTitle,
  DialogContent
} from '@mui/material';
import { useState, useRef, useEffect } from 'react';
import { useSnackDispatch } from '../state/SnackContext';
import { useVideoData, useVideoDataDispatch, METADATA_FIELD_MAPPING } from '../state/VideoDataContext'

/* Expected props:
  none
*/
function FileModal() {

  const videoData = useVideoData();
  const dispatchSnack = useSnackDispatch();
  const dispatchVideoData = useVideoDataDispatch();
  
  const [ modalOpen, setModalOpen ] = useState(true);

  const inputDirectory = useRef(undefined);
  const framesFiles = useRef(undefined);
  const metadataFile = useRef(undefined);
  const annotationsFile = useRef(undefined);
  const videoFile = useRef(undefined);

  const handleDirSelect = (e) => {
    inputDirectory.current = [...e.target.files]
    console.log(inputDirectory.current)
  }

  const handleSubmitClick = (e) => {
    // console.time('Submit Timer')
    if (inputDirectory.current === undefined || inputDirectory.current.length === 0) {
      dispatchSnack({
        type: 'pushSnack', 
        severity: 'warning',
        message: 'You must select an input directory to continue'
      })
      return;
    }


    if (findFiles() === false) {
      return;
    }
    setModalOpen(false);
    // set loading == true
    processInputFiles();
    // console.timeEnd('Submit Timer')

  }

  const resetInput = () => {
    inputDirectory.current = undefined;
    framesFiles.current = undefined;
    metadataFile.current = undefined;
    annotationsFile.current = undefined;
    videoFile.current = undefined;
    setModalOpen(true);
  }

  
  const EXPECTED_DIR_FORMAT = {
    'metadata': {
      'name': 'metadata.json',
      'ext': '.json',
      'altNames': [],
      'altExt': [],
      'relativePath':'./',
      'altPaths': []
    },
    'video': {
      'name': 'decorated_video_bbox_only.mp4',
      'ext': '.mp4',
      'altNames': ['decorated_video', 'video'],
      'altExt': [],
      'relativePath':'./',
      'altPaths': []
    },
    'annotations': {
      'name': 'labels.txt',
      'ext': '.txt',
      'altNames': [],
      'altExt': ['csv'],
      'relativePath':'./',
      'altPaths': ['./raw_data/']
    },
    'frames': {
      // 'name': /frame_(\d*)(\.|_)/,
      'ext': '.jpg',
      'altNames': [],
      'altExt': ['jpeg', 'png'],
      'relativePath':'./',
      'altPaths': ['./raw_data/'],
    }
  }

  // TODO: implement more flexible file finding using EXPECTED_DIR_FORMAT 
  // TODO: improve validation, replace "[].find" approach with more robust approach
  const findFiles = () => {
    let files = [...inputDirectory.current]
    framesFiles.current = files.filter(f =>
      f.webkitRelativePath.toLowerCase().includes('decorated_frames') &&
      f.name.toLowerCase().match(/frame_\d+\./) !== null
    );
    metadataFile.current = files.find(f =>  f.name.toLowerCase().includes('metadata.json'));
    annotationsFile.current = files.find(f =>  f.name.toLowerCase().includes('labels.txt'));
    videoFile.current = files.find(f =>  f.name.toLowerCase().includes('decorated_video_bbox_only.mp4'));
    
    let validInput = true;
    if (framesFiles.current.length === 0) {
      dispatchSnack({
        type:"pushSnack",
        severity:"error",
        message:`Your input directory is missing frames`
      });
      validInput = false
    }
    let INDEX_MAP = ['metadata', 'annotations', 'video'];
    [metadataFile.current, annotationsFile.current, videoFile.current].forEach((a, i) => {
      if(a === undefined){
        dispatchSnack({
          type:"pushSnack",
          severity:"error",
          message:`Your input directory is missing ${INDEX_MAP[i]}`
        });
        validInput = false
      }
    });
    return validInput;
  }

  const processInputFiles = () => {
    processMetadataFile();
    processAnnotationsFile();
    loadVideoFrames();
  }

  const processMetadataFile = () => {
    
    let parsedFile;
    const reader = new FileReader();

    reader.addEventListener('load', (event) => {
      parsedFile = JSON.parse(event.target.result);
      let validMetadata = true
      let tempMetadata = {}
      if (parsedFile === undefined) {
        dispatchSnack({
          type: "pushSnack",
          severity: "error",
          message: `Metadata.json file is empty`
        });
        validMetadata = false;
      } else {
        Object.keys(METADATA_FIELD_MAPPING).forEach((key) => {
            if (parsedFile[METADATA_FIELD_MAPPING[key]] === undefined) {
              validMetadata = false;
              dispatchSnack({
                type: "pushSnack",
                severity: "error",
                message: `Metadata.json missing required key "${METADATA_FIELD_MAPPING[key]}". Please fix to continue.`
              });
            } else { tempMetadata[key] = parsedFile[METADATA_FIELD_MAPPING[key]] }
        });
      }
      // console.log("parsed Metadata", tempMetadata)
      if (validMetadata) {
        dispatchVideoData({
          type: "setMetadata",
          metadata: tempMetadata
        });
      }
    });
    reader.readAsText(metadataFile.current);
  }

  const processAnnotationsFile = () => {

    let parsedFile;
    const reader = new FileReader();

    reader.addEventListener('load', (event) => {
      parsedFile = event.target.result;
      if (parsedFile === undefined || parsedFile === '') {
        dispatchSnack({
          type: "pushSnack",
          severity: "error",
          message: `Annotations file is empty`
        });
      } else {
        let tempAnnotations = {}
        let lines = parsedFile.split('\n')
        lines.forEach((line, i) => {
          let data = line.split(',');
          if (data.length < 3) {
            if(i !== lines.length-1) {
              dispatchSnack({
                type: "pushSnack",
                severity: "error",
                message: `Annotations file has unexpected format "${line}" at line ${i}.`
              });
            }
            return;
          }
          tempAnnotations[data[0]] = {
            machineLabel: data[1],
            confidence: Number(data[2])
          }
        })
        dispatchVideoData({
          type: "setAnnotations",
          annotations: tempAnnotations
        })
        console.log("Annotations:", tempAnnotations)
      }
    });
    reader.readAsText(annotationsFile.current);
  }

  const loadVideoFrames = () => {
    let tempFrames = framesFiles.current.map((file) => {
      let frame = {
        src: URL.createObjectURL(file),
        frameNumber: Number(file.name.match(/\d+/)[0])
      }
      return frame;
    });
    // console.time('Sort time')
    tempFrames = tempFrames.sort((a,b) => a.frameNumber - b.frameNumber);
    // console.timeEnd('Sort time')
    console.log("file", tempFrames[5])
    dispatchVideoData({
      type:"setFrames",
      frames: tempFrames
    });
    // console.log("Exit loading video frames: ", tempFrames );
    // downloadState.current.timer = setTimeout(loadVideoFrame, 1)
  }

  return (
    <Dialog className="FileModal" open={modalOpen}>
      <DialogTitle>Choose Video Directory</DialogTitle>
        <DialogContent>
          {/* <label for="fileInput">Choose project directory</label> */}
          <input 
            type="file" 
            id="fileInput" 
            onChange={handleDirSelect}
            webkitdirectory=""
          />
          {/* <p>No directory selected</p> */}
          <button onClick={handleSubmitClick}>Submit</button>
        </DialogContent>
    </Dialog>
  );
}

export default FileModal;