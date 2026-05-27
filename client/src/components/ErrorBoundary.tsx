/**
 * ErrorBoundary — catches render-time crashes in the routed page tree so a
 * single faulty component shows a recoverable fallback instead of blanking the
 * whole app. Class component because error boundaries have no hook equivalent.
 */

import { Component, type ErrorInfo, type ReactNode } from 'react';
import { AlertTriangle } from 'lucide-react';
import { Button } from './ui/Button';
import './ErrorBoundary.css';

interface ErrorBoundaryProps {
  children: ReactNode;
  /** Optional custom fallback; receives the error and a reset callback. */
  fallback?: (error: Error, reset: () => void) => ReactNode;
}

interface ErrorBoundaryState {
  error: Error | null;
}

export class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  state: ErrorBoundaryState = { error: null };

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    // Surface to the console for diagnosis; a telemetry sink can hook in here.
    console.error('[ErrorBoundary] render crash:', error, info.componentStack);
  }

  reset = () => this.setState({ error: null });

  render() {
    const { error } = this.state;
    if (!error) return this.props.children;

    if (this.props.fallback) return this.props.fallback(error, this.reset);

    return (
      <div className="error-boundary" role="alert">
        <div className="error-boundary__card">
          <span className="error-boundary__icon"><AlertTriangle size={32} /></span>
          <h2 className="error-boundary__title">Something broke on this screen</h2>
          <p className="error-boundary__message">{error.message || 'An unexpected error occurred.'}</p>
          <div className="error-boundary__actions">
            <Button variant="secondary" onClick={this.reset}>Try again</Button>
            <Button variant="primary" onClick={() => window.location.reload()}>Reload app</Button>
          </div>
        </div>
      </div>
    );
  }
}
