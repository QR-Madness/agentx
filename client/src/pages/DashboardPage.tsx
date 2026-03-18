/**
 * DashboardPage — System overview with health, metrics, and storage stats
 */

import {
  Activity,
  Server,
  Brain,
  Wrench,
  Database,
  RefreshCw,
  CheckCircle2,
  AlertCircle,
  XCircle,
  HardDrive,
  MessageSquare,
  Layers,
} from 'lucide-react';
import {
  useHealth,
  useProviders,
  useMCPServers,
  useAgentStatus,
  useMemoryStats,
} from '../lib/hooks';
import { useServer } from '../contexts/ServerContext';
import { useConversation } from '../contexts/ConversationContext';
import './DashboardPage.css';

export function DashboardPage() {
  const { activeServer } = useServer();
  const { data: health, loading: healthLoading, error: healthError, refresh: refreshHealth } = useHealth();
  const { providers, loading: providersLoading } = useProviders();
  const { servers: mcpServers, loading: mcpLoading } = useMCPServers();
  const { status: agentStatus } = useAgentStatus();
  const { stats: memoryStats, loading: memoryLoading } = useMemoryStats();
  const { serverConversations } = useConversation();

  const getStatusIcon = (status: string | undefined) => {
    switch (status) {
      case 'healthy':
        return <CheckCircle2 size={16} className="status-icon healthy" />;
      case 'degraded':
        return <AlertCircle size={16} className="status-icon degraded" />;
      default:
        return <XCircle size={16} className="status-icon unhealthy" />;
    }
  };

  // Format bytes to human readable
  const formatBytes = (mb: number | undefined) => {
    if (mb === undefined) return '—';
    if (mb < 1) return `${(mb * 1024).toFixed(0)} KB`;
    if (mb < 1024) return `${mb.toFixed(1)} MB`;
    return `${(mb / 1024).toFixed(2)} GB`;
  };

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
        <button className="button-ghost" onClick={refreshHealth}>
          <RefreshCw size={16} />
        </button>
      </div>

      {/* System Status Grid - Row 1 */}
      <div className="status-grid">
        {/* System Health */}
        <div className="status-card card">
          <div className="status-header">
            <div className="status-title">
              <Activity size={18} className="status-title-icon" />
              <span>System Health</span>
            </div>
            {healthLoading ? (
              <div className="shimmer status-badge-placeholder"></div>
            ) : healthError ? (
              <div className="status-badge unhealthy">
                <XCircle size={14} />
                <span>Offline</span>
              </div>
            ) : (
              <div className={`status-badge ${health?.status}`}>
                {getStatusIcon(health?.status)}
                <span>{health?.status}</span>
              </div>
            )}
          </div>
          <div className="status-details">
            <div className="detail-row">
              <span className="detail-label">API</span>
              <span className={`detail-status ${health?.api?.status === 'healthy' ? 'online' : 'offline'}`}>
                {health?.api?.status || 'Unknown'}
              </span>
            </div>
            <div className="detail-row">
              <span className="detail-label">Translation</span>
              <span className={`detail-status ${health?.translation?.status === 'healthy' ? 'online' : 'offline'}`}>
                {health?.translation?.status || 'Unknown'}
              </span>
            </div>
          </div>
        </div>

        {/* Database Status */}
        <div className="status-card card">
          <div className="status-header">
            <div className="status-title">
              <Database size={18} className="status-title-icon" />
              <span>Databases</span>
            </div>
          </div>
          <div className="status-details">
            <div className="detail-row">
              <span className="detail-label">Neo4j</span>
              <span className={`detail-status ${health?.memory?.neo4j?.status === 'healthy' ? 'online' : 'offline'}`}>
                {health?.memory?.neo4j?.status || 'Unknown'}
              </span>
            </div>
            <div className="detail-row">
              <span className="detail-label">PostgreSQL</span>
              <span className={`detail-status ${health?.memory?.postgres?.status === 'healthy' ? 'online' : 'offline'}`}>
                {health?.memory?.postgres?.status || 'Unknown'}
              </span>
            </div>
            <div className="detail-row">
              <span className="detail-label">Redis</span>
              <span className={`detail-status ${health?.memory?.redis?.status === 'healthy' ? 'online' : 'offline'}`}>
                {health?.memory?.redis?.status || 'Unknown'}
              </span>
            </div>
          </div>
        </div>

        {/* Model Providers */}
        <div className="status-card card">
          <div className="status-header">
            <div className="status-title">
              <Brain size={18} className="status-title-icon" />
              <span>Model Providers</span>
            </div>
            <span className="status-count">
              {providersLoading ? '...' : `${providers.filter(p => p.available).length}/${providers.length}`}
            </span>
          </div>
          <div className="status-details">
            {providersLoading ? (
              <div className="shimmer detail-placeholder"></div>
            ) : providers.length === 0 ? (
              <p className="no-data">No providers configured</p>
            ) : (
              providers.slice(0, 3).map(provider => (
                <div key={provider.name} className="detail-row">
                  <span className="detail-label">{provider.name}</span>
                  <span className={`detail-status ${provider.available ? 'online' : 'offline'}`}>
                    {provider.available ? 'Available' : 'Unavailable'}
                  </span>
                </div>
              ))
            )}
          </div>
        </div>

        {/* MCP Servers */}
        <div className="status-card card">
          <div className="status-header">
            <div className="status-title">
              <Wrench size={18} className="status-title-icon" />
              <span>MCP Servers</span>
            </div>
            <span className="status-count">
              {mcpLoading ? '...' : mcpServers.length}
            </span>
          </div>
          <div className="status-details">
            {mcpLoading ? (
              <div className="shimmer detail-placeholder"></div>
            ) : mcpServers.length === 0 ? (
              <p className="no-data">No MCP servers connected</p>
            ) : (
              mcpServers.slice(0, 3).map(server => (
                <div key={server.name} className="detail-row">
                  <span className="detail-label">{server.name}</span>
                  <span className={`detail-status ${server.status === 'connected' ? 'online' : 'offline'}`}>
                    {server.status}
                  </span>
                </div>
              ))
            )}
          </div>
        </div>
      </div>

      {/* Metrics Grid - Row 2 */}
      <div className="status-grid">
        {/* Memory Stats */}
        <div className="status-card card">
          <div className="status-header">
            <div className="status-title">
              <Layers size={18} className="status-title-icon" />
              <span>Memory Stats</span>
            </div>
          </div>
          <div className="status-details">
            {memoryLoading ? (
              <div className="shimmer detail-placeholder"></div>
            ) : !memoryStats ? (
              <p className="no-data">Unable to load stats</p>
            ) : (
              <>
                <div className="detail-row">
                  <span className="detail-label">Entities</span>
                  <span className="detail-value">{memoryStats.totals.entities.toLocaleString()}</span>
                </div>
                <div className="detail-row">
                  <span className="detail-label">Facts</span>
                  <span className="detail-value">{memoryStats.totals.facts.toLocaleString()}</span>
                </div>
                <div className="detail-row">
                  <span className="detail-label">Turns</span>
                  <span className="detail-value">{memoryStats.totals.turns.toLocaleString()}</span>
                </div>
              </>
            )}
          </div>
        </div>

        {/* Storage Metrics */}
        <div className="status-card card">
          <div className="status-header">
            <div className="status-title">
              <HardDrive size={18} className="status-title-icon" />
              <span>Storage</span>
            </div>
          </div>
          <div className="status-details">
            {healthLoading ? (
              <div className="shimmer detail-placeholder"></div>
            ) : !health?.storage ? (
              <>
                <div className="detail-row">
                  <span className="detail-label">PostgreSQL</span>
                  <span className="detail-value muted">—</span>
                </div>
                <div className="detail-row">
                  <span className="detail-label">Neo4j</span>
                  <span className="detail-value muted">—</span>
                </div>
                <div className="detail-row">
                  <span className="detail-label">Redis</span>
                  <span className="detail-value muted">—</span>
                </div>
              </>
            ) : (
              <>
                <div className="detail-row">
                  <span className="detail-label">PostgreSQL</span>
                  <span className="detail-value">{formatBytes(health.storage.postgres_size_mb)}</span>
                </div>
                <div className="detail-row">
                  <span className="detail-label">Neo4j</span>
                  <span className="detail-value">{formatBytes(health.storage.neo4j_size_mb)}</span>
                </div>
                <div className="detail-row">
                  <span className="detail-label">Redis</span>
                  <span className="detail-value">{formatBytes(health.storage.redis_memory_mb)}</span>
                </div>
              </>
            )}
          </div>
        </div>

        {/* Active Conversations */}
        <div className="status-card card">
          <div className="status-header">
            <div className="status-title">
              <MessageSquare size={18} className="status-title-icon" />
              <span>Conversations</span>
            </div>
            <span className="status-count">{serverConversations.length}</span>
          </div>
          <div className="status-details">
            <div className="detail-row">
              <span className="detail-label">Saved</span>
              <span className="detail-value">{serverConversations.length}</span>
            </div>
            <div className="detail-row">
              <span className="detail-label">Channels</span>
              <span className="detail-value">
                {new Set(serverConversations.map(c => c.channel)).size}
              </span>
            </div>
          </div>
        </div>

        {/* Agent Status */}
        <div className="status-card card">
          <div className="status-header">
            <div className="status-title">
              <Activity size={18} className="status-title-icon" />
              <span>Agent Status</span>
            </div>
            {agentStatus && (
              <div className={`status-badge ${agentStatus.status === 'ready' ? 'healthy' : 'degraded'}`}>
                {agentStatus.status}
              </div>
            )}
          </div>
          <div className="status-details">
            <div className="detail-row">
              <span className="detail-label">Active Sessions</span>
              <span className="detail-value">{agentStatus?.active_sessions ?? 0}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
