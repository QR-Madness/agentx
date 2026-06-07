import { ServerProvider } from './contexts/ServerContext';
import { AuthProvider } from './contexts/AuthContext';
import { ThemeProvider } from './contexts/ThemeContext';
import { ModalProvider } from './contexts/ModalContext';
import { ConversationProvider } from './contexts/ConversationContext';
import { AmbassadorProvider } from './contexts/AmbassadorContext';
import { PlansProvider } from './contexts/PlansContext';
import { UIChromeProvider } from './contexts/UIChromeContext';
import { AgentProfileProvider } from './contexts/AgentProfileContext';
import { AlloyWorkflowProvider } from './contexts/AlloyWorkflowContext';
import { NotificationProvider } from './contexts/NotificationContext';
import { TooltipProvider } from './components/ui/Tooltip';
import { Toaster } from './components/ui/Toaster';
import { ConfirmProvider } from './components/ui/ConfirmDialog';
import { ModalPortal } from './components/modals/ModalPortal';
import { ErrorBoundary } from './components/ErrorBoundary';
import { RootLayout } from './layouts/RootLayout';
import { ResizeHandles } from './layouts/ResizeHandles';
import { showWindowControls } from './lib/platform';
import './App.css';

/**
 * Self-contained fallback for the outermost boundary. It must render even when
 * the crash came from a provider above it (e.g. ThemeProvider) — so it uses only
 * inline styles and no theme tokens, app components, or context. Surfaces the
 * stack so a blank-screen startup crash is debuggable on-device (no DevTools).
 */
function RootErrorFallback(error: Error) {
  return (
    <div
      role="alert"
      style={{
        position: 'fixed',
        inset: 0,
        overflow: 'auto',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        gap: 16,
        padding: 24,
        background: '#05070f',
        color: '#e2e8f0',
        font: '14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif',
      }}
    >
      <h1 style={{ margin: 0, fontSize: 20, fontWeight: 600 }}>AgentX failed to start</h1>
      <p style={{ margin: 0, maxWidth: 560, textAlign: 'center', color: '#94a3b8' }}>
        {error.message || 'An unexpected error occurred during startup.'}
      </p>
      {error.stack && (
        <pre
          style={{
            maxWidth: '100%',
            maxHeight: '40vh',
            overflow: 'auto',
            margin: 0,
            padding: 12,
            borderRadius: 8,
            background: 'rgba(255,255,255,0.06)',
            color: '#cbd5e1',
            fontSize: 12,
            whiteSpace: 'pre-wrap',
            wordBreak: 'break-word',
          }}
        >
          {error.stack}
        </pre>
      )}
      <button
        type="button"
        onClick={() => window.location.reload()}
        style={{
          padding: '8px 18px',
          borderRadius: 8,
          border: '1px solid rgba(255,255,255,0.16)',
          background: '#8b5cf6',
          color: '#fff',
          fontSize: 14,
          fontWeight: 600,
          cursor: 'pointer',
        }}
      >
        Reload app
      </button>
    </div>
  );
}

function App() {
  return (
    <ErrorBoundary fallback={(error) => RootErrorFallback(error)}>
      <ServerProvider>
        <AuthProvider>
          <ThemeProvider>
            <NotificationProvider>
              <TooltipProvider>
                <AgentProfileProvider>
                  <AlloyWorkflowProvider>
                    <ConversationProvider>
                      <AmbassadorProvider>
                       <PlansProvider>
                        <ModalProvider>
                          <UIChromeProvider>
                            <ConfirmProvider>
                              <ErrorBoundary>
                                <RootLayout />
                              </ErrorBoundary>
                              <ModalPortal />
                              <Toaster />
                              {showWindowControls && <ResizeHandles />}
                            </ConfirmProvider>
                          </UIChromeProvider>
                        </ModalProvider>
                       </PlansProvider>
                      </AmbassadorProvider>
                    </ConversationProvider>
                  </AlloyWorkflowProvider>
                </AgentProfileProvider>
              </TooltipProvider>
            </NotificationProvider>
          </ThemeProvider>
        </AuthProvider>
      </ServerProvider>
    </ErrorBoundary>
  );
}

export default App;
