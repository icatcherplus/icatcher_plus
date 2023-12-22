import {
  Typography
} from '@mui/material'
import { useEffect, useState } from 'react';
import { useVideoData } from '../../state/VideoDataProvider';
import HeatmapCanvas from './HeatmapCanvas';

import styles from './AnnotationsBar.module.css'

function AnnotationsBar(props) {
  
  const { id, getColorArray } = props;
  const videoData = useVideoData();
  const [ colorArray, setColorArray ] = useState([]);

  useEffect(()=> {
    videoData.annotations[id].length > 0 
      ? setColorArray(getColorArray())
      : setColorArray([]);
  },[videoData.annotations, id])  // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div className={styles.temp}>
      <div className={styles.text}>
        <Typography
          align="right"
          noWrap
          variant="button"
        >
          {`${idToName(id)}:`}
        </Typography>
      </div>
      <HeatmapCanvas colorArray={colorArray}/>

    </div>
  );
}
  
export default AnnotationsBar;

const idToName = (id) => {
  return id.split(/(?=[A-Z])/).join(" ")
}