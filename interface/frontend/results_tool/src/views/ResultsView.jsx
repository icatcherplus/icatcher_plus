import { VideoDataProvider } from '../state/VideoDataProvider';
import { PlaybackStateProvider } from '../state/PlaybackStateProvider';

import Instructions from '../components/Instructions';
import FileModal from '../components/FileModal';
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
          <FileModal />
        </div>
      </PlaybackStateProvider>
    </VideoDataProvider>
  );
}

export default ResultsView;
