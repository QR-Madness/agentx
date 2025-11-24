import React from 'react';
import '../../styles/DashboardTab.css';

export const DashboardTab: React.FC = () => {
  const quickActions = [
    { icon: '🌐', title: 'Quick Translation', description: 'Translate text instantly', action: 'translation' },
    { icon: '💬', title: 'Start Chat', description: 'Begin a new conversation', action: 'chat' },
    { icon: '📁', title: 'File Analysis', description: 'Analyze documents', action: 'tools' },
    { icon: '🔧', title: 'Tools', description: 'Access utility tools', action: 'tools' },
  ];

  const stats = [
    { label: 'Translations', value: '1,234', trend: '+12%', icon: '📊' },
    { label: 'Conversations', value: '89', trend: '+5%', icon: '💭' },
    { label: 'Files Analyzed', value: '456', trend: '+8%', icon: '📈' },
  ];

  return (
    <div className="dashboard-tab">
      <div className="dashboard-hero fade-in">
        <h1 className="hero-title">
          Welcome to <span className="gradient-text">AgentX</span>
        </h1>
        <p className="hero-subtitle">
          Your powerful AI toolbox for translation, chat, and more
        </p>
      </div>

      <div className="dashboard-stats">
        {stats.map((stat, index) => (
          <div key={index} className="stat-card card slide-in" style={{ animationDelay: `${index * 0.1}s` }}>
            <div className="stat-icon">{stat.icon}</div>
            <div className="stat-content">
              <div className="stat-label">{stat.label}</div>
              <div className="stat-value">{stat.value}</div>
              <div className="stat-trend success">{stat.trend}</div>
            </div>
          </div>
        ))}
      </div>

      <div className="quick-actions">
        <h2 className="section-title">Quick Actions</h2>
        <div className="actions-grid">
          {quickActions.map((action, index) => (
            <button
              key={index}
              className="action-card glass"
              style={{ animationDelay: `${index * 0.1 + 0.2}s` }}
            >
              <div className="action-icon">{action.icon}</div>
              <div className="action-title">{action.title}</div>
              <div className="action-description">{action.description}</div>
            </button>
          ))}
        </div>
      </div>

      <div className="recent-activity card">
        <h2 className="section-title">Recent Activity</h2>
        <div className="activity-list">
          <div className="activity-item">
            <div className="activity-icon">🌐</div>
            <div className="activity-content">
              <div className="activity-title">Translated document</div>
              <div className="activity-time">2 minutes ago</div>
            </div>
          </div>
          <div className="activity-item">
            <div className="activity-icon">💬</div>
            <div className="activity-content">
              <div className="activity-title">New conversation started</div>
              <div className="activity-time">15 minutes ago</div>
            </div>
          </div>
          <div className="activity-item">
            <div className="activity-icon">📁</div>
            <div className="activity-content">
              <div className="activity-title">File analyzed</div>
              <div className="activity-time">1 hour ago</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
