import { useState, useRef, useEffect } from 'react';
import { useVideoData, useVideoDataDispatch } from '../../state/VideoDataProvider';
import styles from './HeatmapCanvas.module.css';

  
/* Expected props:
colorArray: frame image to display
handleClick: callback to handle click on canvas
*/
function HeatmapCanvas(props) {
  
  const { colorArray, handleClick, width } = props;
  const videoData = useVideoData();

  const canvasRef = useRef();


  useEffect (() => {
    if (canvasRef.current !== undefined) {
      canvasRef.current.width = width
      if (colorArray.length !== 0) {
        console.log('painting heatmap')
        paintCanvas(canvasRef, colorArray, videoData)
      }
      else {
        clearCanvas();
      }
      // ? paintCanvas(canvasRef.current, colorArray, videoData)
      // : clearCanvas();
    }
  },[width, colorArray, canvasRef, videoData])
  

  const clearCanvas = () => {
    if (canvasRef.current === undefined) { return }
    const context = canvasRef.current.getContext('2d');
    context.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
  }

  colorArray.length !== 0
    ? paintCanvas(canvasRef, colorArray, videoData)
    : clearCanvas();
  
  return (
    <canvas 
      id="videoCanvas"
      ref={canvasRef}
      className={styles.videoCanvas}
      style={{width: width}}
      onClick={handleClick}
    />
  );
}
  
export default HeatmapCanvas;

const paintCanvas = (canvasRef, colorArray, videoData) => {
  if (canvasRef.current === undefined) { return }
  const context = canvasRef.current.getContext('2d')
  let totalWidth = canvasRef.current.offsetWidth;
  let sliceWidth = totalWidth / videoData.metadata.numFrames;
  let sliceHeight = canvasRef.current.height;

  colorArray.forEach((color, frameNumber) => {
    let x = sliceWidth * (frameNumber - videoData.metadata.frameOffset - 1)
    context.fillStyle = color;
    console.log("Paintin slice at x: "+ x +", y: " + 0 + ", fw: " + sliceWidth + ", fh: " + sliceHeight);
    context.fillRect(x , 0 , sliceWidth, sliceHeight);
  })
}
