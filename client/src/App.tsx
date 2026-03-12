import { ServerProvider } from './contexts/ServerContext';
import { ThemeProvider } from './contexts/ThemeContext';
import { ModalProvider } from './contexts/ModalContext';
import { ModalPortal } from './components/modals/ModalPortal';
import { RootLayout } from './layouts/RootLayout';
import './App.css';

function App() {
  return (
    <ServerProvider>
      <ThemeProvider>
        <ModalProvider>
          <RootLayout />
          <ModalPortal />
        </ModalProvider>
      </ThemeProvider>
    </ServerProvider>
  );
}

export default App;
