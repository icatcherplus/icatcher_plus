import { useEffect, useState } from 'react';
import { addSnack, useSnacksDispatch } from '../../state/SnacksProvider';
import { useVideoData, useVideoDataDispatch } from '../../state/VideoDataProvider';
import HeatmapCanvas from './HeatmapCanvas';


/* Expected props:
tbd
*/

const colorPalettes = {
  categorical: {
    machineLabel: [
      '#C5C5C5',
      '#F05039',
      '#1F449C',
      '#EEBAB4',
      '#7CA1CC'
    ],
    edited: [],
    default: [

    ],
  },
  continuous: {
    confidence: {
      '0%': '#FFFFFF',
      '100%': '#000000'
    },
    default: {
      '0%': '#FFFFFF',
      '100%': '#000000'
    }
  },
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

function AnnotationBar(props) {
  
  const { id, type, totalWidth } = props;
  const videoData = useVideoData();

  const [ colorArray, setColorArray ] = useState([]);
  const [ colorPalette, setColorPalette ] = useState(id);

  useEffect(()=> {
    videoData.annotations[id].length > 0 ?
      computeColorArray()
    :
      setColorArray([]);

  },[videoData.annotations, id])

  const computeColorArray = () => {
    let tempColorArray = []
    const palette = colorPalettes[type][colorPalette || 'default']
    console.log("palette", palette)
    if (type === 'categorical') {
      let categories = [ ...new Set(videoData.annotations[id])]
      let colorMap = {}
      categories.forEach((l, i) => colorMap[l] = palette[i])
      console.log(id, categories)
      videoData.annotations[id].forEach((a, i) => {
        tempColorArray[i] = colorMap[a]
      })
    }
    if (type === 'continuous') {
      //TODO: get min/max, convert to percentages
      videoData.annotations[id].forEach((a, i) => {
        let rgb = blendColors(hexToRgb(palette['0%']), hexToRgb(palette['100%']), a)
        tempColorArray[i] = 'rgb(' + rgb.r + ', ' + rgb.g + ', ' + rgb.b + ')';
      })
    }
    setColorArray([...tempColorArray])
  }


  return (
    <div width={totalWidth} >
      <HeatmapCanvas colorArray={colorArray} width={totalWidth}/>
    </div>
  );
}
  
export default AnnotationBar;