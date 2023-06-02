import { 
  Dialog,
  DialogTitle,
  DialogContent
} from '@mui/material';
import { useRef, useEffect } from 'react';
import styles from './VideoCanvas.module.css';
import { useSnackDispatch } from '../../state/SnackContext';
import { useVideoData, useVideoDataDispatch } from '../../state/VideoDataContext';

  
/* Expected props:
frameToDraw: frame image to display
*/
function VideoCanvas(props) {
  
  const { frameToDraw } = props;

  const canvasRef = useRef();

  console.log("loaded canvas: ", frameToDraw)
  const paintCanvas = () => {
    const context = canvasRef.current.getContext('2d')
    context.drawImage(frameToDraw, 0, 0);
    console.log("Image shown: " + frameToDraw.frameNumber);
  }

  if (frameToDraw !== undefined) {
    paintCanvas();
  }

  return (
    <div>
      <canvas 
        id="videoCanvas" 
        width="854" 
        height="480"
        ref={canvasRef}
        className={styles.videoCanvas}
      >
      </canvas>
    </div>
  );
}
  
export default VideoCanvas;