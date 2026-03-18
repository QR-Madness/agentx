/**
 * RootLayout — Main app shell with top bar and page content area
 */

import { useState, useEffect, useRef } from 'react';
import { TopBar, PageId } from './TopBar';
import { StartPage } from '../pages/StartPage';
import { DashboardPage } from '../pages/DashboardPage';
import { AgentXPage } from '../pages/AgentXPage';
import './RootLayout.css';

export function RootLayout() {
  const [activePage, setActivePage] = useState<PageId>('agentx');
  const [cursorPos, setCursorPos] = useState({ x: 50, y: 50 });
  const rafRef = useRef<number | null>(null);

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
          <StartPage />
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
