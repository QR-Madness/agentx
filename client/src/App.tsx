import { ServerProvider } from './contexts/ServerContext';
import { ThemeProvider } from './contexts/ThemeContext';
import { ModalProvider } from './contexts/ModalContext';
import { ConversationProvider } from './contexts/ConversationContext';
import { ModalPortal } from './components/modals/ModalPortal';
import { RootLayout } from './layouts/RootLayout';
import './App.css';

function App() {
  return (
    <ServerProvider>
      <ThemeProvider>
        <ConversationProvider>
          <ModalProvider>
            <RootLayout />
            <ModalPortal />
          </ModalProvider>
        </ConversationProvider>
      </ThemeProvider>
    </ServerProvider>
  );
}

export default App;
