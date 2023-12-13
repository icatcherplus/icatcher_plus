import { useRef, useEffect } from 'react';
import { usePlaybackState } from '../../state/PlaybackStateProvider';
import styles from './VideoCanvas.module.css';
  
/* Expected props:
frameToDraw: frame image to display
handleClick: callback to handle click on canvas
*/
function VideoCanvas(props) {
  
  const { frameToDraw, handleClick, handleKeyDown } = props;
  const playbackState = usePlaybackState();
  const canvasRef = useRef();

  useEffect (() => {
    if (canvasRef.current !== undefined) { 
      canvasRef.current.width = playbackState.videoWidth
      canvasRef.current.height = playbackState.videoWidth * (1/playbackState.aspectRatio)  
      frameToDraw !== undefined
        ? paintCanvas(canvasRef, playbackState.videoWidth, playbackState.aspectRatio, frameToDraw)
        : clearCanvas(canvasRef);
    }
  },[playbackState.videoWidth, playbackState.aspectRatio, frameToDraw])
  
  return (
    <canvas 
      id="videoCanvas"
      ref={canvasRef}
      className={styles.videoCanvas}
      onClick={handleClick}
      onKeyDown = {(e) => onKeyDown(e, canvasRef, handleKeyDown)}
      tabIndex={0}
    />
  );
}
  
export default VideoCanvas;

const paintCanvas = (canvasRef, width, aspectRatio, frameToDraw) => {
  const context = canvasRef.current.getContext('2d')
  context.drawImage(frameToDraw, 0, 0, width, width*(1/aspectRatio));
}

const clearCanvas = (canvasRef) => {
  const context = canvasRef.current.getContext('2d');
  context.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
}

const onKeyDown = (event, canvasRef, handleKeyDown) => {
  if (canvasRef.current === undefined) { return }
  handleKeyDown(event)
}