import React, { useState } from 'react';
import {
  Wrench,
  Server,
  Search,
  Play,
  ChevronRight,
  ExternalLink,
  Code,
  Database,
  FileText,
  Globe,
  RefreshCw,
  Plug,
  Unplug,
  Loader2
} from 'lucide-react';
import { useMCPServers, useMCPTools } from '../../lib/hooks';
import '../../styles/ToolsPanel.css';

export const ToolsPanel: React.FC = () => {
  const { servers, loading: serversLoading, refresh: refreshServers, connectServer, connectAll, disconnectServer } = useMCPServers();
  const { tools, loading: toolsLoading, refresh: refreshTools } = useMCPTools();
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedTool, setSelectedTool] = useState<string | null>(null);
  const [connecting, setConnecting] = useState<string | null>(null);

  const filteredTools = tools.filter(tool =>
    tool.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
    tool.description?.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const getToolIcon = (toolName: string) => {
    if (toolName.includes('file') || toolName.includes('read') || toolName.includes('write')) {
      return <FileText size={18} />;
    }
    if (toolName.includes('database') || toolName.includes('sql') || toolName.includes('query')) {
      return <Database size={18} />;
    }
    if (toolName.includes('http') || toolName.includes('fetch') || toolName.includes('request')) {
      return <Globe size={18} />;
    }
    if (toolName.includes('code') || toolName.includes('execute') || toolName.includes('run')) {
      return <Code size={18} />;
    }
    return <Wrench size={18} />;
  };

  return (
    <div className="tools-tab">
      {/* Header */}
      <div className="tools-header fade-in">
        <div className="header-content">
          <h1 className="page-title">
            <Wrench className="page-icon-svg" />
            <span>Tools</span>
          </h1>
          <p className="page-subtitle">Browse and test MCP tools from connected servers</p>
        </div>
        <button className="button-secondary" onClick={() => { refreshServers(); refreshTools(); }}>
          <RefreshCw size={16} />
          Refresh
        </button>
        {servers.length > 0 && (
          <button 
            className="button-primary"
            onClick={async () => {
              setConnecting('__all__');
              await connectAll();
              await refreshTools();
              setConnecting(null);
            }}
            disabled={connecting !== null}
          >
            {connecting === '__all__' ? <Loader2 size={16} className="spin" /> : <Plug size={16} />}
            Connect All
          </button>
        )}
      </div>

      {/* MCP Servers Overview */}
      <div className="servers-section">
        <h2 className="section-title">
          <Server size={18} className="section-title-icon" />
          Connected MCP Servers
        </h2>
        <div className="servers-grid">
          {serversLoading ? (
            <>
              <div className="server-card card shimmer"></div>
              <div className="server-card card shimmer"></div>
            </>
          ) : servers.length === 0 ? (
            <div className="empty-state card">
              <Server size={32} className="empty-icon" />
              <p>No MCP servers connected</p>
              <span className="empty-hint">Configure servers in Settings → MCP Servers</span>
            </div>
          ) : (
            servers.map(server => (
              <div key={server.name} className="server-card card">
                <div className="server-icon">
                  <Server size={20} />
                </div>
                <div className="server-info">
                  <span className="server-name">{server.name}</span>
                  <span className="server-tools">
                    {server.status === 'connected'
                      ? `${server.tools_count || server.tools?.length || 0} tools available`
                      : server.transport || 'disconnected'}
                  </span>
                </div>
                <div className="server-actions">
                  {server.status === 'connected' ? (
                    <button
                      className="button-ghost button-sm"
                      title="Disconnect"
                      onClick={async () => {
                        setConnecting(server.name);
                        await disconnectServer(server.name);
                        await refreshTools();
                        setConnecting(null);
                      }}
                      disabled={connecting !== null}
                    >
                      {connecting === server.name ? <Loader2 size={14} className="spin" /> : <Unplug size={14} />}
                    </button>
                  ) : (
                    <button
                      className="button-ghost button-sm"
                      title="Connect"
                      onClick={async () => {
                        setConnecting(server.name);
                        await connectServer(server.name);
                        await refreshTools();
                        setConnecting(null);
                      }}
                      disabled={connecting !== null}
                    >
                      {connecting === server.name ? <Loader2 size={14} className="spin" /> : <Plug size={14} />}
                    </button>
                  )}
                </div>
                <div className={`status-dot ${server.status === 'connected' ? 'online' : 'offline'}`}></div>
              </div>
            ))
          )}
        </div>
      </div>

      {/* Tools Browser */}
      <div className="tools-browser">
        <div className="browser-header">
          <h2 className="section-title">
            <Wrench size={18} className="section-title-icon" />
            Available Tools
            {!toolsLoading && <span className="tool-count">{tools.length}</span>}
          </h2>
          <div className="search-wrapper">
            <Search size={18} className="search-icon" />
            <input
              type="text"
              placeholder="Search tools..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="search-input"
            />
          </div>
        </div>

        <div className="tools-grid">
          {toolsLoading ? (
            <>
              <div className="tool-card card shimmer"></div>
              <div className="tool-card card shimmer"></div>
              <div className="tool-card card shimmer"></div>
            </>
          ) : filteredTools.length === 0 ? (
            <div className="empty-state card full-width">
              <Wrench size={32} className="empty-icon" />
              {searchQuery ? (
                <p>No tools match "{searchQuery}"</p>
              ) : (
                <p>No tools available from connected servers</p>
              )}
            </div>
          ) : (
            filteredTools.map(tool => (
              <div 
                key={`${tool.server}-${tool.name}`} 
                className={`tool-card card ${selectedTool === tool.name ? 'selected' : ''}`}
                onClick={() => setSelectedTool(selectedTool === tool.name ? null : tool.name)}
              >
                <div className="tool-header">
                  <div className="tool-icon">
                    {getToolIcon(tool.name)}
                  </div>
                  <div className="tool-info">
                    <span className="tool-name">{tool.name}</span>
                    <span className="tool-server">{tool.server}</span>
                  </div>
                  <ChevronRight size={18} className="tool-arrow" />
                </div>
                <p className="tool-description">{tool.description || 'No description available'}</p>
                
                {selectedTool === tool.name && (
                  <div className="tool-actions">
                    <button className="button-primary">
                      <Play size={14} />
                      Test Tool
                    </button>
                    <button className="button-ghost">
                      <ExternalLink size={14} />
                      View Schema
                    </button>
                  </div>
                )}
              </div>
            ))
          )}
        </div>
      </div>

      {/* Tool Testing Panel (placeholder) */}
      {selectedTool && (
        <div className="testing-panel card fade-in">
          <h3 className="panel-title">Test: {selectedTool}</h3>
          <div className="testing-content">
            <p className="testing-placeholder">
              Tool testing interface coming soon. Select a tool and configure inputs to test it.
            </p>
          </div>
        </div>
      )}
    </div>
  );
};
