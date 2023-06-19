import { 
  Drawer
} from '@mui/material';
import styles from './App.module.css';
import Instructions from './components/Instructions';
import FileModal from './components/FileModal';
import Snacks from './components/common/Snacks';
import VideoFrame from './components/video/VideoFrame';

function App() {

  return (
    <div className={styles.app}>
      <div className={styles.mainpage}>
        <Instructions />
        <VideoFrame />
      </div>
      <FileModal />
      <Snacks />
    </div>
  );
}

export default App;
