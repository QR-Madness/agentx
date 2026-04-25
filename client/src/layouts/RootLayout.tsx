/**
 * RootLayout — Main app shell with top bar and page content area
 */

import { useState, useEffect, useRef } from 'react';
import { Loader2 } from 'lucide-react';
import { TopBar, PageId } from './TopBar';
import { StartPage } from '../pages/StartPage';
import { DashboardPage } from '../pages/DashboardPage';
import { AgentXPage } from '../pages/AgentXPage';
import { AuthPage } from '../pages/AuthPage';
import { VersionMismatchPage } from '../pages/VersionMismatchPage';
import { useConversation } from '../contexts/ConversationContext';
import { useAuth } from '../contexts/AuthContext';
import { useModal } from '../contexts/ModalContext';
import './RootLayout.css';

export function RootLayout() {
  const {
    isAuthenticated,
    authRequired,
    isLoading,
    versionMismatch,
    versionInfo,
    checkAuthStatus,
  } = useAuth();
  const [activePage, setActivePage] = useState<PageId>('start');
  const [cursorPos, setCursorPos] = useState({ x: 50, y: 50 });
  const rafRef = useRef<number | null>(null);
  const { addTab, closeTab, activeTabId } = useConversation();
  const { openModal } = useModal();

  // Global keyboard shortcuts
  useEffect(() => {
    // Skip if not authenticated
    if (isLoading || (authRequired && !isAuthenticated)) return;

    const handleKeyDown = (e: KeyboardEvent) => {
      const mod = e.metaKey || e.ctrlKey;
      if (!mod) return;

      if (e.key === 't') {
        e.preventDefault();
        addTab();
        setActivePage('agentx');
      } else if (e.key === 'w') {
        e.preventDefault();
        if (activeTabId) closeTab(activeTabId);
      } else if (e.key === ',') {
        e.preventDefault();
        // Open unified settings (Cmd+,)
        openModal({
          id: 'unified-settings',
          type: 'modal',
          component: 'unifiedSettings',
          size: 'full',
        });
      } else if (e.key === 'k') {
        e.preventDefault();
        // Command palette — placeholder for future implementation
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [addTab, closeTab, activeTabId, openModal, isLoading, authRequired, isAuthenticated]);

  // Track cursor position for reactive gradient
  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (rafRef.current) return; // Throttle with rAF
      rafRef.current = requestAnimationFrame(() => {
        const x = (e.clientX / window.innerWidth) * 100;
        const y = (e.clientY / window.innerHeight) * 100;
        setCursorPos({ x, y });
        rafRef.current = null;
      });
    };

    window.addEventListener('mousemove', handleMouseMove);
    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
  }, []);

  // Show loading screen while checking auth
  if (isLoading) {
    return (
      <div className="root-layout root-layout--loading">
        <div className="loading-spinner">
          <Loader2 size={32} className="animate-spin" />
          <span>Connecting...</span>
        </div>
      </div>
    );
  }

  // Show version mismatch page if versions are incompatible
  if (versionMismatch) {
    return (
      <div
        className="root-layout"
        style={{
          '--cursor-x': `${cursorPos.x}%`,
          '--cursor-y': `${cursorPos.y}%`,
        } as React.CSSProperties}
      >
        <VersionMismatchPage versionInfo={versionInfo} onRetry={checkAuthStatus} />
      </div>
    );
  }

  // Show auth page if auth is required and not authenticated
  if (authRequired && !isAuthenticated) {
    return (
      <div
        className="root-layout"
        style={{
          '--cursor-x': `${cursorPos.x}%`,
          '--cursor-y': `${cursorPos.y}%`,
        } as React.CSSProperties}
      >
        <AuthPage />
      </div>
    );
  }

  return (
    <div
      className="root-layout"
      style={{
        '--cursor-x': `${cursorPos.x}%`,
        '--cursor-y': `${cursorPos.y}%`,
      } as React.CSSProperties}
    >
      <TopBar activePage={activePage} onPageChange={setActivePage} />

      <main className="page-content">
        {/* All pages always mounted to preserve state; visibility toggled via CSS */}
        <div
          className="page-wrapper"
          style={{ display: activePage === 'start' ? 'block' : 'none' }}
        >
          <StartPage onNavigate={setActivePage} />
        </div>
        <div
          className="page-wrapper"
          style={{ display: activePage === 'dashboard' ? 'block' : 'none' }}
        >
          <DashboardPage />
        </div>
        <div
          className="page-wrapper"
          style={{ display: activePage === 'agentx' ? 'block' : 'none' }}
        >
          <AgentXPage />
        </div>
      </main>
    </div>
  );
}
