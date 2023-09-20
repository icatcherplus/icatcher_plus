import { VideoDataProvider } from '../state/VideoDataProvider';
import { PlaybackStateProvider } from '../state/PlaybackStateProvider';

import Instructions from '../components/Instructions';
import UploadModal from '../components/UploadModal';
import VideoFrame from '../components/video/VideoFrame';
import AnnotationsFrame from '../components/annotations/AnnotationsFrame';

import styles from './ResultsView.module.css';

function ResultsView() {

  return (
    <VideoDataProvider>
      <PlaybackStateProvider>
        <div className={styles.mainpage}>
          <Instructions />
          <div className={styles.content} >
            <VideoFrame />
            <AnnotationsFrame />  
          </div>
          <UploadModal />
        </div>
      </PlaybackStateProvider>
    </VideoDataProvider>
  );
}

export default ResultsView;
