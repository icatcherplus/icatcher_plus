import { 
  Dialog,
  DialogTitle,
  DialogContent
} from '@mui/material';
import { useState, useRef, useEffect } from 'react';
import styles from './VideoCanvas.module.css';

  
/* Expected props:
frameToDraw: frame image to display
handleClick: callback to handle click on canvas
*/
function VideoCanvas(props) {
  
  const { frameToDraw, handleClick, handleKeyDown, width, aspectRatio } = props;

  const canvasRef = useRef();


  useEffect (() => {
    if (canvasRef.current !== undefined) { 
      console.log('setting dimensions')
      canvasRef.current.width = width
      canvasRef.current.height = width * (1/aspectRatio)  
      frameToDraw !== undefined
      ? paintCanvas()
      : clearCanvas();
    }
  },[width, aspectRatio])

  const paintCanvas = () => {
    console.log("painting canvas", frameToDraw)
    if (canvasRef.current === undefined) { return }
    const context = canvasRef.current.getContext('2d')
    context.drawImage(frameToDraw, 0, 0, width, width * (1/aspectRatio));
  }

  const clearCanvas = () => {
    console.log("clearing canvas")
    if (canvasRef.current === undefined) { return }
    const context = canvasRef.current.getContext('2d');
    context.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
  }

  const onKeyDown = (e) => {
    if (canvasRef.current === undefined) { return }
    handleKeyDown(e)
  }

  console.log("rendering canvas")
  frameToDraw !== undefined
    ? paintCanvas()
    : clearCanvas();
  
  return (
    <canvas 
      id="videoCanvas"
      ref={canvasRef}
      className={styles.videoCanvas}
      onClick={handleClick}
      onKeyDown = {onKeyDown}
      tabIndex={0}
    />
  );
}
  
export default VideoCanvas;