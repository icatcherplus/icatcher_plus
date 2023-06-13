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
handleClick: callback to handle click on canvas
*/
function VideoCanvas(props) {
  
  const { frameToDraw, handleClick, handleKeyDown } = props;

  const canvasRef = useRef();
  const paintCanvas = () => {
    if (canvasRef.current === undefined) { return }
    // console.log("Showing image", frameToDraw.frameNumber)
    canvasRef.current.width = frameToDraw.width;
    canvasRef.current.height = frameToDraw.height;
    const context = canvasRef.current.getContext('2d')
    context.drawImage(frameToDraw, 0, 0);
  }

  const clearCanvas = () => {
    if (canvasRef.current === undefined) { return }
    const context = canvasRef.current.getContext('2d');
    context.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
  }

  const onKeyDown = (e) => {
    if (canvasRef.current === undefined) { return }
    handleKeyDown(e)
  }

  frameToDraw !== undefined
    ? paintCanvas()
    : clearCanvas();

  
  return (
    <div>
      <canvas 
        id="videoCanvas" 
        width="854" 
        height="480"
        ref={canvasRef}
        className={styles.videoCanvas}
        onClick={handleClick}
        onKeyDown = {onKeyDown}
        tabIndex={0}
      >
      </canvas>
    </div>
  );
}
  
export default VideoCanvas;