import React from 'react';
import {
  LayoutDashboard,
  Bot,
  Languages,
  MessageSquare,
  Wrench,
  Settings,
  Sparkles,
  Database
} from 'lucide-react';
import '../styles/TabBar.css';

export type TabId = 'dashboard' | 'agent' | 'translation' | 'chat' | 'tools' | 'memory' | 'settings';

export interface Tab {
  id: TabId;
  label: string;
  icon: React.ReactNode;
}

interface TabBarProps {
  tabs: Tab[];
  activeTab: TabId;
  onTabChange: (tabId: TabId) => void;
}

export const TabBar: React.FC<TabBarProps> = ({ tabs, activeTab, onTabChange }) => {
  return (
    <div className="tab-bar">
      <div className="tab-bar-header">
        <div className="app-logo">
          <div className="logo-icon">
            <Sparkles size={24} />
          </div>
          <span className="logo-text gradient-text">AgentX</span>
        </div>
      </div>

      <nav className="tab-nav">
        {tabs.map(tab => (
          <button
            key={tab.id}
            className={`tab-button ${activeTab === tab.id ? 'active' : ''}`}
            onClick={() => onTabChange(tab.id)}
          >
            <span className="tab-icon">{tab.icon}</span>
            <span className="tab-label">{tab.label}</span>
            {activeTab === tab.id && <div className="tab-indicator" />}
          </button>
        ))}
      </nav>

      <div className="tab-bar-footer">
        <div className="server-status">
          <span className="status-dot online"></span>
          <span className="server-label">Connected</span>
        </div>
      </div>
    </div>
  );
};

// Export icons for use in App.tsx
export const TabIcons = {
  dashboard: <LayoutDashboard size={20} />,
  agent: <Bot size={20} />,
  translation: <Languages size={20} />,
  chat: <MessageSquare size={20} />,
  tools: <Wrench size={20} />,
  memory: <Database size={20} />,
  settings: <Settings size={20} />,
};
