/**
 * RootLayout — Main app shell with top bar and page content area
 */

import { useState } from 'react';
import { TopBar, PageId } from './TopBar';
import { StartPage } from '../pages/StartPage';
import { DashboardPage } from '../pages/DashboardPage';
import { AgentXPage } from '../pages/AgentXPage';
import './RootLayout.css';

export function RootLayout() {
  const [activePage, setActivePage] = useState<PageId>('agentx');

  return (
    <div className="root-layout">
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
