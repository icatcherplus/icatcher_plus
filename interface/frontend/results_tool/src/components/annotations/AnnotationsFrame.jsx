import { 
  Skeleton
} from '@mui/material';
import { useEffect, useState, useRef } from 'react';
import { useSnacksDispatch, addSnack } from '../../state/SnacksProvider';
import { useVideoData, useVideoDataDispatch } from '../../state/VideoDataProvider';
import AnnotationBar from './AnnotationBar';
import styles from './AnnotationsFrame.module.css';
  
/* Expected props:
  tbd
*/
function AnnotationsFrame(props) {

  const { width } = props;
  const videoData = useVideoData();
  const dispatchSnack = useSnacksDispatch();
  
  const [ percentLoaded, setPercentLoaded ] = useState(0);

//   useEffect(() => {
//     // var elem = document.getElementById('myPos');
//     // var w = document.getElementById('myFrameDownloadProgress').offsetWidth;

//     // //Calculate x pos from current frame
//     // var offsetxPos = (m_play_state.current_frame / (m_manifest.video.num_frames - 1)) * w;

//     // elem.style.left = offsetxPos;
// }, [currentFrame])

  useEffect(() => {
    let loadedFrames = videoData.frames.length;
    if (loadedFrames > 0) {
      if (videoData.metadata.numFrames <= 0) {
        addSnack(
          'Video metadata incorrectly lists video as 0 frames long',
          "error",
          dispatchSnack
        )
      }
      else {
        let l = (loadedFrames - videoData.frameOffset - 1)/videoData.metadata.numFrames * 100.0
        // if (l > percentLoaded) {
          setPercentLoaded(l)
        // }
      }
    }
  }, [
    videoData.frames.length, 
    videoData.frameOffset,
    videoData.metadata.numFrames,
    dispatchSnack
  ])

  // const getAnnotationBar = (key) => {
  // //   let dataArray = videoData.annotations[key]
  // //   console.log('getting data', dataArray, key)
  //   return <AnnotationBar width={width} id={key} />
  // }
  
  console.log("Showing heatmap bar")

  return (
    <div className={styles.annotationsBar}>
      {
        Object.keys(videoData.annotations).length !== 0 ? 
          Object.keys(videoData.annotations).map((key) => {
            return <AnnotationBar 
              totalWidth={width} 
              // currentFrame={currentFrame} 
              id={key} 
              type={ key==='confidence' ? 'continuous':'categorical' } 
            />
            // return <div id={key} className={styles.visible}/>
          })
          : 
          <Skeleton 
            variant="text" 
            width={width} 
            height={100} 
          />
      }
    </div>
  );
}

export default AnnotationsFrame;