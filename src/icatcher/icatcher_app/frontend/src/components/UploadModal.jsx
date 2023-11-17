import { 
  Dialog,
  DialogTitle,
  DialogContent,
  DialogContentText
} from '@mui/material';
import { useState, useRef } from 'react';
import { useSnacksDispatch, addSnack } from '../state/SnacksProvider';
import { useVideoDataDispatch, addFrames } from '../state/VideoDataProvider'
import UploadButton from './UploadButton'

/* Expected props:
  none
*/
function UploadModal() {

  const dispatchSnack = useSnacksDispatch();
  const dispatchVideoData = useVideoDataDispatch();
  
  const [ modalOpen, setModalOpen ] = useState(true);

  const inputDirectory = useRef(undefined);
  const framesFiles = useRef(undefined);
  const annotationsFile = useRef(undefined);

  const handleDirectoryUpload = (e) => {
    inputDirectory.current = [...e.target.files]
  }

  const handleSubmitClick = (e) => {
    if (inputDirectory.current === undefined || inputDirectory.current.length === 0) {
      dispatchSnack(addSnack('You must select an input directory to continue', 'warning'))
      return;
    }


    if (findFiles() === false) {
      return;
    }
    setModalOpen(false);
    processInputFiles();

  }

  const findFiles = () => {
    let files = [...inputDirectory.current]
    framesFiles.current = files.filter(f =>
      f.webkitRelativePath.toLowerCase().includes('decorated_frames') &&
      f.name.toLowerCase().match(/frame_\d+\./) !== null
    );
    annotationsFile.current = files.find(f =>  f.name.toLowerCase().includes('labels.txt'));
    
    let validInput = true;
    if (framesFiles.current.length === 0) {
      dispatchSnack(addSnack(`Your input directory is missing frames`, "error"))
      validInput = false
    }
    if(annotationsFile.current === undefined){
      dispatchSnack(addSnack(`Your input directory is missing annotations`, "error"))
      validInput = false
    }
    return validInput;
  }

  const processInputFiles = () => {
    processAnnotationsFile();
    loadVideoFrames();
  }

  const processAnnotationsFile = () => {

    let parsedFile;
    const reader = new FileReader();

    reader.addEventListener('load', (event) => {
      parsedFile = event.target.result;
      if (parsedFile === undefined || parsedFile === '') {
        dispatchSnack(addSnack(`Annotations file is empty`, "error"))
      } else {
        let tempAnnotations = {
          machineLabel: [],
          confidence: []
        }
        let lines = parsedFile.split('\n')
        lines.forEach((line, i) => {
          let data = line.split(',');
          if (data.length < 3) {
            if(i !== lines.length-1) {
              dispatchSnack(addSnack(
                `Annotations file has unexpected format "${line}" at line ${i}.`, 
                "error"
              ))
            }
            return;
          }
          let index = Number(data[0])
          tempAnnotations.machineLabel[index] = data[1].trim()
          tempAnnotations.confidence[index] = Number(data[2])
        })
        dispatchVideoData({
          type: "setAnnotations",
          annotations: tempAnnotations
        })
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
    tempFrames = tempFrames.sort((a,b) => a.frameNumber - b.frameNumber);
    dispatchVideoData(addFrames(tempFrames))
  }

  return (
    <Dialog open={modalOpen} >
      <DialogTitle>
        Choose Video Directory
        <DialogContentText component={'div'} style={{minWidth: '30vw'}} >
          Directory should include:
          <ul style={{margin: '0px'}}>
            <li>annotations as label.txt</li>
            <li>directory containing video frames</li>
          </ul>
        </DialogContentText>
      </DialogTitle>
      
      <DialogContent>
        <div style={{display: 'flex', justifyContent: 'space-between'}}>
          <UploadButton handleInput={handleDirectoryUpload} />
          <button onClick={handleSubmitClick}>Submit</button>
        </div>
      </DialogContent>
    </Dialog>
  );
}

export default UploadModal;