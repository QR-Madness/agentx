import { ServerProvider } from './contexts/ServerContext';
import { AuthProvider } from './contexts/AuthContext';
import { ThemeProvider } from './contexts/ThemeContext';
import { ModalProvider } from './contexts/ModalContext';
import { ConversationProvider } from './contexts/ConversationContext';
import { AgentProfileProvider } from './contexts/AgentProfileContext';
import { AlloyWorkflowProvider } from './contexts/AlloyWorkflowContext';
import { TooltipProvider } from './components/ui/Tooltip';
import { ModalPortal } from './components/modals/ModalPortal';
import { RootLayout } from './layouts/RootLayout';
import './App.css';

function App() {
  return (
    <ServerProvider>
      <AuthProvider>
        <ThemeProvider>
          <TooltipProvider>
            <AgentProfileProvider>
              <AlloyWorkflowProvider>
                <ConversationProvider>
                  <ModalProvider>
                    <RootLayout />
                    <ModalPortal />
                  </ModalProvider>
                </ConversationProvider>
              </AlloyWorkflowProvider>
            </AgentProfileProvider>
          </TooltipProvider>
        </ThemeProvider>
      </AuthProvider>
    </ServerProvider>
  );
}

export default App;
