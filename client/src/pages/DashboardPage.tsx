/**
 * DashboardPage — Clean, mean, powerful.
 *
 * Slim health strip up top (SystemStatusStrip) + Usage & Cost hero
 * (UsageMetricsSection). Subsystem detail lives behind the strip's
 * Details toggle; all the same hooks fire under the hood.
 */

import { Server } from 'lucide-react';
import { useServer } from '../contexts/ServerContext';
import { SystemStatusStrip } from '../components/dashboard/SystemStatusStrip';
import { UsageMetricsSection } from '../components/dashboard/UsageMetricsSection';
import './DashboardPage.css';

export function DashboardPage() {
  const { activeServer } = useServer();

  return (
    <div className="dashboard-page">
      {/* Server Connection Banner */}
      <div className="server-banner card glass">
        <div className="banner-content">
          <Server size={20} className="banner-icon" />
          <div className="banner-info">
            <span className="banner-label">Connected to</span>
            <span className="banner-value">{activeServer?.name || 'No server'}</span>
          </div>
          <span className="banner-url">{activeServer?.url}</span>
        </div>
      </div>

      <SystemStatusStrip />
      <UsageMetricsSection />
    </div>
  );
}
