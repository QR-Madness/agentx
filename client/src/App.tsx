import { ServerProvider } from './contexts/ServerContext';
import { AuthProvider } from './contexts/AuthContext';
import { ThemeProvider } from './contexts/ThemeContext';
import { ModalProvider } from './contexts/ModalContext';
import { ConversationProvider } from './contexts/ConversationContext';
import { PlansProvider } from './contexts/PlansContext';
import { AgentProfileProvider } from './contexts/AgentProfileContext';
import { AlloyWorkflowProvider } from './contexts/AlloyWorkflowContext';
import { NotificationProvider } from './contexts/NotificationContext';
import { TooltipProvider } from './components/ui/Tooltip';
import { Toaster } from './components/ui/Toaster';
import { ModalPortal } from './components/modals/ModalPortal';
import { ErrorBoundary } from './components/ErrorBoundary';
import { RootLayout } from './layouts/RootLayout';
import './App.css';

function App() {
  return (
    <ServerProvider>
      <AuthProvider>
        <ThemeProvider>
          <NotificationProvider>
            <TooltipProvider>
              <AgentProfileProvider>
                <AlloyWorkflowProvider>
                  <ConversationProvider>
                    <PlansProvider>
                      <ModalProvider>
                        <ErrorBoundary>
                          <RootLayout />
                        </ErrorBoundary>
                        <ModalPortal />
                        <Toaster />
                      </ModalProvider>
                    </PlansProvider>
                  </ConversationProvider>
                </AlloyWorkflowProvider>
              </AgentProfileProvider>
            </TooltipProvider>
          </NotificationProvider>
        </ThemeProvider>
      </AuthProvider>
    </ServerProvider>
  );
}

export default App;
