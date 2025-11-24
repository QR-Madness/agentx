import React from 'react';
import '../../styles/ToolsTab.css';

export const ToolsTab: React.FC = () => {
  const tools = [
    {
      icon: '📝',
      title: 'Text Analysis',
      description: 'Analyze text sentiment and structure',
      status: 'available',
    },
    {
      icon: '📄',
      title: 'Document Parser',
      description: 'Extract data from documents',
      status: 'available',
    },
    {
      icon: '🔍',
      title: 'Code Analyzer',
      description: 'Review and analyze code quality',
      status: 'coming-soon',
    },
    {
      icon: '🎨',
      title: 'Image Tools',
      description: 'Process and analyze images',
      status: 'coming-soon',
    },
    {
      icon: '📊',
      title: 'Data Visualizer',
      description: 'Create charts and graphs',
      status: 'coming-soon',
    },
    {
      icon: '🔐',
      title: 'Encryption Tools',
      description: 'Encrypt and decrypt data',
      status: 'available',
    },
  ];

  return (
    <div className="tools-tab">
      <div className="tools-header fade-in">
        <h1 className="page-title">
          <span className="page-icon">🔧</span>
          Tools & Utilities
        </h1>
        <p className="page-subtitle">Access powerful AI-powered tools and utilities</p>
      </div>

      <div className="tools-grid">
        {tools.map((tool, index) => (
          <div
            key={index}
            className={`tool-card card ${tool.status}`}
            style={{ animationDelay: `${index * 0.1}s` }}
          >
            <div className="tool-icon">{tool.icon}</div>
            <div className="tool-info">
              <h3 className="tool-title">{tool.title}</h3>
              <p className="tool-description">{tool.description}</p>
            </div>
            {tool.status === 'available' ? (
              <button className="tool-button button-primary">
                Launch →
              </button>
            ) : (
              <div className="coming-soon-badge">Coming Soon</div>
            )}
          </div>
        ))}
      </div>

      <div className="tools-info card">
        <h2 className="section-title">About Tools</h2>
        <p className="info-text">
          AgentX provides a comprehensive suite of AI-powered tools to enhance your productivity.
          Each tool is designed to work seamlessly with the others, providing a unified experience.
        </p>
        <div className="info-stats">
          <div className="info-stat">
            <span className="stat-number">6</span>
            <span className="stat-label">Total Tools</span>
          </div>
          <div className="info-stat">
            <span className="stat-number">3</span>
            <span className="stat-label">Available Now</span>
          </div>
          <div className="info-stat">
            <span className="stat-number">3</span>
            <span className="stat-label">Coming Soon</span>
          </div>
        </div>
      </div>
    </div>
  );
};
