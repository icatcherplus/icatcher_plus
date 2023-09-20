import { 
  Skeleton
} from '@mui/material';
import { useVideoData } from '../../state/VideoDataProvider';
import { usePlaybackState } from '../../state/PlaybackStateProvider';

import ContinuousAnnotationBar from './ContinuousAnnotationBar';
import CategoricalAnnotationBar from './CategoricalAnnotationBar';
import AnnotationsScrubBar from './AnnotationsScrubBar';

import styles from './AnnotationsFrame.module.css';
  

function AnnotationsFrame(props) {

  const videoData = useVideoData();
  const playbackState = usePlaybackState();
  
  
  
  return (
    <div className={styles.annotationsBar}>
      {
        Object.keys(videoData.annotations).length !== 0 ? 
          <div>
            <AnnotationsScrubBar />
            { 
              Object.keys(videoData.annotations).map((key) => {
                return key==='confidence'
                  ? <ContinuousAnnotationBar 
                      key={key} 
                      id={key}
                      palette={'confidence'}
                    />
                  : <CategoricalAnnotationBar 
                      key={key} 
                      id={key}
                    />
              })
            }
          </div>
          : 
          <Skeleton 
            variant="text" 
            width={playbackState.videoWidth}
            height={100} 
          />
      }
    </div>
  );
}

export default AnnotationsFrame;