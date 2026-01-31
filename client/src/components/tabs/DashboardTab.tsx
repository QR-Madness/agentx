import React from 'react';
import {
  Activity,
  Server,
  Brain,
  Zap,
  Wrench,
  Database,
  RefreshCw,
  ArrowRight,
  Sparkles,
  CheckCircle2,
  AlertCircle,
  XCircle
} from 'lucide-react';
import { useHealth, useProviders, useMCPServers, useAgentStatus } from '../../lib/hooks';
import { useServer } from '../../contexts/ServerContext';
import '../../styles/DashboardTab.css';

export const DashboardTab: React.FC = () => {
  const { activeServer } = useServer();
  const { data: health, loading: healthLoading, error: healthError, refresh: refreshHealth } = useHealth();
  const { providers, loading: providersLoading } = useProviders();
  const { servers: mcpServers, loading: mcpLoading } = useMCPServers();
  const { status: agentStatus } = useAgentStatus();

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

  return (
    <div className="dashboard-tab">
      {/* Hero Section */}
      <div className="dashboard-hero fade-in">
        <div className="hero-content">
          <h1 className="hero-title">
            Welcome to <span className="gradient-text">AgentX</span>
          </h1>
          <p className="hero-subtitle">
            AI Agent Platform for orchestration, reasoning, and tool execution
          </p>
        </div>
        <div className="hero-glow"></div>
      </div>

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

      {/* System Status Grid */}
      <div className="status-grid">
        {/* API Health */}
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

      {/* Quick Actions */}
      <div className="quick-actions-section">
        <h2 className="section-title">
          <Zap size={20} className="section-title-icon" />
          Quick Actions
        </h2>
        <div className="actions-grid">
          <button className="action-card glass">
            <div className="action-icon-wrapper">
              <Brain size={24} />
            </div>
            <div className="action-content">
              <span className="action-title">Run Agent Task</span>
              <span className="action-description">Execute an AI-powered task</span>
            </div>
            <ArrowRight size={18} className="action-arrow" />
          </button>

          <button className="action-card glass">
            <div className="action-icon-wrapper">
              <Sparkles size={24} />
            </div>
            <div className="action-content">
              <span className="action-title">Quick Translation</span>
              <span className="action-description">Translate text instantly</span>
            </div>
            <ArrowRight size={18} className="action-arrow" />
          </button>

          <button className="action-card glass">
            <div className="action-icon-wrapper">
              <Wrench size={24} />
            </div>
            <div className="action-content">
              <span className="action-title">Browse Tools</span>
              <span className="action-description">Explore available MCP tools</span>
            </div>
            <ArrowRight size={18} className="action-arrow" />
          </button>
        </div>
      </div>

      {/* Agent Status */}
      <div className="agent-status-section card">
        <div className="agent-status-header">
          <h2 className="section-title">
            <Activity size={20} className="section-title-icon" />
            Agent Status
          </h2>
          {agentStatus && (
            <div className={`status-badge ${agentStatus.status === 'ready' ? 'healthy' : 'degraded'}`}>
              {agentStatus.status}
            </div>
          )}
        </div>
        <div className="agent-metrics">
          <div className="metric">
            <span className="metric-value">{agentStatus?.active_sessions ?? 0}</span>
            <span className="metric-label">Active Sessions</span>
          </div>
          <div className="metric">
            <span className="metric-value">0</span>
            <span className="metric-label">Tasks Today</span>
          </div>
          <div className="metric">
            <span className="metric-value">0</span>
            <span className="metric-label">Tokens Used</span>
          </div>
        </div>
      </div>
    </div>
  );
};
