import React from 'react';
import '../styles/TabBar.css';

export type TabId = 'dashboard' | 'translation' | 'chat' | 'tools';

export interface Tab {
  id: TabId;
  label: string;
  icon: string;
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
        <div className="app-title">
          <span className="app-icon">✨</span>
          <span className="gradient-text">AgentX</span>
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
        <button className="settings-button button-secondary">
          <span>⚙️</span>
          <span>Settings</span>
        </button>
      </div>
    </div>
  );
};
