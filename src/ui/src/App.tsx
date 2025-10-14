import { Provider } from 'react-redux';
import { store } from './store/store';
import Terminal from './components/Layout/Terminal';
import './index.css';

function App() {
  return (
    <Provider store={store}>
      <div className="h-screen w-screen bg-terminal-bg text-terminal-text font-terminal overflow-hidden">
        <Terminal />
      </div>
    </Provider>
  );
}

export default App;