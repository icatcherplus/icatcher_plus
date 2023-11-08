import { useRef, useEffect } from 'react';
import { useVideoData } from '../../state/VideoDataProvider';
import { usePlaybackState } from '../../state/PlaybackStateProvider';
import styles from './HeatmapCanvas.module.css';

  
/* Expected props:
colorArray: frame image to display
handleClick: callback to handle click on canvas
*/
function HeatmapCanvas(props) {
  
  const { colorArray, handleClick } = props;
  const videoData = useVideoData();
  const playbackState = usePlaybackState();


  const canvasRef = useRef();


  useEffect (() => {
    if (canvasRef.current !== undefined && Object.keys(videoData.metadata).length !== 0) {
      canvasRef.current.width = playbackState.videoWidth
      if (colorArray.length !== 0) {
        paintCanvas(canvasRef, colorArray, videoData)
      }
      else {
        clearCanvas();
      }
    }
  },[playbackState.videoWidth, colorArray, canvasRef, videoData])
  

  const clearCanvas = () => {
    if (canvasRef.current === undefined) { return }
    const context = canvasRef.current.getContext('2d');
    context.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
  }
  
  return (
    <canvas 
      id="videoCanvas"
      ref={canvasRef}
      className={styles.videoCanvas}
      style={{width: playbackState.videoWidth}}
      onClick={handleClick}
    />
  );
}
  
export default HeatmapCanvas;

const paintCanvas = (canvasRef, colorArray, videoData) => {
  if (canvasRef.current === undefined) { return }
  const context = canvasRef.current.getContext('2d')
  let totalWidth = canvasRef.current.offsetWidth;
  let sliceWidth = totalWidth / (videoData.metadata.numFrames - 8);
  let sliceHeight = canvasRef.current.height;

  colorArray.forEach((color, frameNumber) => {
    let x = sliceWidth * frameNumber
    context.fillStyle = "black";
    context.fillStyle = color;
    context.fillRect(Math.ceil(x) , 0 , Math.ceil(sliceWidth), sliceHeight);
  })
}
