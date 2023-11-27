import { VideoDataProvider } from '../state/VideoDataProvider';
import { PlaybackStateProvider } from '../state/PlaybackStateProvider';

import Instructions from '../components/Instructions';
import UploadModal from '../components/UploadModal';
import VideoFrame from '../components/video/VideoFrame';
import AnnotationsFrame from '../components/annotations/AnnotationsFrame';
import FrameInfoFrame from '../components/frameinfo/FrameInfoFrame';
import FilterBoxFrame from '../components/filterbox/FilterBoxFrame';
import EditorBox from '../components/editor/EditorBox'
// import JumpToFrame from '../components/jumptoframe/JumpToFrame';

import styles from './ResultsView.module.css';

function ResultsView() {

  return (
    <VideoDataProvider>
      <PlaybackStateProvider>
        <div className={styles.mainpage}>
          <Instructions />
          <div className={styles.content}>

            <div className={styles.left} >
{/*               <h1> left </h1> */}
              <FrameInfoFrame />
{/*               <JumpToFrame /> */}
            </div>

            <div className={styles.center} >
{/*               <h1> center </h1> */}
              <VideoFrame />
              <AnnotationsFrame />
            </div>

            <div className={styles.right} >
{/*               <h1> right </h1> */}
              <EditorBox />
              <FilterBoxFrame />
            </div>

          </div>
          <UploadModal />
        </div>
      </PlaybackStateProvider>
    </VideoDataProvider>
  );
}

export default ResultsView;
