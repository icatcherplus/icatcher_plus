import styles from './App.module.css';
import FileModal from './components/FileModal';
import Snacks from './components/common/Snacks';
import VideoFrame from './components/video/VideoFrame';

function App() {

  return (
    <div className={styles.app}>
      <div className={styles.mainpage}>
        <div className={styles.instructionContainer}>
          <h4 className={styles.instructionsHeader}>Welcome to iCatcher+</h4>
          <p className={styles.instructions}>Click the video or press SPACE to start play</p>
          <p className={styles.instructions}>Use the 'r' key to play the video in reverse</p>
          <p className={styles.instructions}>Use the left and right arrow keys to page through the video frames</p>
          <p className={styles.instructions}>Jump to a specific frame using the input box above the video</p>
        </div>
        <VideoFrame />
      </div>
      <FileModal />
      <Snacks />
    </div>
  );
}

export default App;
