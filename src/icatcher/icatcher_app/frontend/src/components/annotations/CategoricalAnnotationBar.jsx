import { useEffect, useState } from 'react';
import { useVideoData } from '../../state/VideoDataProvider';
import AnnotationBar from './AnnotationsBar';
import styles from './ContinuousAnnotationBar.module.css'

const colorPalettes = {
  machineLabel: {
    'undefined': '#C5C5C5',
    'left': '#F05039',
    'right': '#1F449C',
    'away': '#EEBAB4',
    'noface': '#7CA1CC',
    'nobabyface': '#000000'
  },
  edited: {
    'edited': '#F05039',
    'unedited':'#FFFFFF'
  },
  default: {
    '0': '#C5C5C5',
    '1': '#F05039',
    '2': '#1F449C',
    '3': '#EEBAB4',
    '4': '#7CA1CC',
    '5': '#000000'
  }
}

function CategoricalAnnotationBar(props) {
  
  const { id } = props;
  const videoData = useVideoData();

  const [ colorPalette, setColorPalette ] = useState(id);
  const [ selectedLabel, setSelectedLabel ] = useState('away');
  const [ labelOptions, setLabelOptions ] = useState([])

  useEffect (()=> {
    let labels = [ ...new Set(videoData.annotations[id])].map((l) => {return String(l)})
    setLabelOptions(labels)
    setSelectedLabel(labels[0])
    setColorPalette(id)
    
  }, [id, videoData.annotations])  // eslint-disable-line react-hooks/exhaustive-deps

  const computeColorArray = () => {
    let tempColorArray = []
    const palette = colorPalettes[colorPalette || 'default']
    const annotationArray = videoData.annotations[id]
    annotationArray.forEach((a, i) => {
      tempColorArray[i] = palette[`${a}`]
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
  
export default CategoricalAnnotationBar;
