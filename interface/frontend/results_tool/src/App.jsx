import styles from './App.module.css';
import Instructions from './components/Instructions';
import FileModal from './components/FileModal';
import Snacks from './components/common/Snacks';
import VideoFrame from './components/video/VideoFrame';
import AnnotationsFrame from './components/annotations/AnnotationsFrame';

function App() {

  return (
    <div className={styles.app}>
      <div className={styles.mainpage}>
        <Instructions />
        <VideoFrame />
        <AnnotationsFrame />
      </div>
      <FileModal />
      <Snacks />
    </div>
  );
}

export default App;
