import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import App from './App';
import reportWebVitals from './reportWebVitals';
import { SnacksProvider } from './state/SnacksProvider';
import { ViewProvider } from './state/ViewProvider'

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <SnacksProvider>
      <ViewProvider>
        <App />
      </ViewProvider>
    </SnacksProvider>
  </React.StrictMode>
);

reportWebVitals(console.log);
