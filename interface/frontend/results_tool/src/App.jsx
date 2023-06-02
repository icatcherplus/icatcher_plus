import styles from './App.module.css';
import FileModal from './components/FileModal';
import Snacks from './components/common/Snacks';
import VideoFrame from './components/video/VideoFrame';

function App() {

  return (
    <div className={styles.app}>
      <div className={styles.mainpage}>
        <div className={styles.instructionContainer}>
          <h4 className={styles.instructions}>An Instruction Heading</h4>
          <p className={styles.instructions}>The instructions</p>
        </div>
        <VideoFrame />
      </div>
      <FileModal />
      <Snacks />
    </div>
  );
}

export default App;
