import { useView } from './state/ViewProvider'
import UploadView from './views/UploadView';
import ProgressView from './views/ProgressView';
import ResultsView from './views/ResultsView';
import Snacks from './components/common/Snacks';

import styles from './App.module.css';


const VIEWS_MAP = {
  UPLOAD: <UploadView />,
  PROGRESS: <ProgressView />,
  RESULTS: <ResultsView />
}

function App() {

  const viewState = useView();

  return (
    <div className={styles.app}>
      { VIEWS_MAP[viewState?.currentView] }
      <Snacks />
    </div>
  );
}

export default App;
