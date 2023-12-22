import { useEffect, useState } from 'react';
import { useVideoData } from '../../state/VideoDataProvider';
import AnnotationBar from './AnnotationsBar';

import styles from './ContinuousAnnotationBar.module.css'

/* Expected props:
  id:string --> key for annotation array in videoData.annotations
  palette:string --> key for color palette from colorPalettes object below
*/

const colorPalettes = {
  confidence: {
    '0%': '#FFFFFF',
    '100%': '#000000'
  },
  monochromeRed: {
    '0%': '#EEBAB4',
    '100%': '#F05039'
  },
  monochromeBlue: {
    '0%': '#7CA1CC',
    '100%': '#1F449C'
  },
  default: {
    '0%': '#FFFFFF',
    '100%': '#000000'
  }
}

function ContinuousAnnotationBar(props) {
  
  const { id } = props;
  const videoData = useVideoData();

  const [ threshold, setThreshold ] = useState(0.80);
  const [ range, setRange ] = useState({ min: 0, max: 1})
  const [ annotationArray, setAnnotationArray ] = useState(videoData.annotations[id]);


  useEffect (()=> {
    let [ newMin, newMax ] = getRange()
    setRange({min: newMin, max: newMax})
    if(newMin > threshold || newMax < threshold) {
      setThreshold((newMax-newMin)/2)
    }
    setAnnotationArray(videoData.annotations[id])
  }, [id, videoData.annotations])  // eslint-disable-line react-hooks/exhaustive-deps

  const computeColorArray = () => {
    let tempColorArray = []
    const palette = colorPalettes[id || 'default']
    annotationArray.forEach((a, i) => {
      let rgb = blendColors(hexToRgb(palette['0%']), hexToRgb(palette['100%']), normalizeValue(a, range))
      tempColorArray[i] = 'rgb(' + rgb.r + ', ' + rgb.g + ', ' + rgb.b + ')';
    })
    return [...tempColorArray]
  }

  return (
    <div className={styles.temp}>
      <AnnotationBar
        id={id}
        getColorArray={computeColorArray}
      >
      </AnnotationBar>
    </div>
  );
}
  
export default ContinuousAnnotationBar;

const normalizeValue = (value, range) => {
  //get value normalized between minimum and maximum 
  return (value-range.min)/(range.max-range.min)
}

const getRange = () => {
  const min = 0
  const max = 1
  return [min, max]
}

const hexToRgb = (hex) => {
  return {
      r: parseInt(hex.substring(1, 3), 16),
      g: parseInt(hex.substring(3, 5), 16),
      b: parseInt(hex.substring(5, 7), 16)
  };
}

const blendColors = (colorA, colorB, weight) => {
  return {
      r: Math.floor(colorA.r * (1 - weight) + colorB.r * weight),
      g: Math.floor(colorA.g * (1 - weight) + colorB.g * weight),
      b: Math.floor(colorA.b * (1 - weight) + colorB.b * weight)
  };
}