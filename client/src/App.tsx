import { ServerProvider } from './contexts/ServerContext';
import { ThemeProvider } from './contexts/ThemeContext';
import { ModalProvider } from './contexts/ModalContext';
import { ConversationProvider } from './contexts/ConversationContext';
import { AgentProfileProvider } from './contexts/AgentProfileContext';
import { ModalPortal } from './components/modals/ModalPortal';
import { RootLayout } from './layouts/RootLayout';
import './App.css';

function App() {
  return (
    <ServerProvider>
      <ThemeProvider>
        <AgentProfileProvider>
          <ConversationProvider>
            <ModalProvider>
              <RootLayout />
              <ModalPortal />
            </ModalProvider>
          </ConversationProvider>
        </AgentProfileProvider>
      </ThemeProvider>
    </ServerProvider>
  );
}

export default App;
