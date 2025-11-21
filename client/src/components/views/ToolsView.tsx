import React from 'react';
import styled from 'styled-components';

const ViewContainer = styled.div`
  display: flex;
  flex-direction: column;
  height: 100%;
  padding: 24px;
`;

const ViewHeader = styled.div`
  margin-bottom: 24px;

  h2 {
    font-size: 24px;
    font-weight: 600;
    color: ${({ theme }) => theme.colors.textPrimary};
  }
`;

const ToolsContainer = styled.div`
  height: 100%;
  overflow-y: auto;
`;

const ToolsGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
  gap: 20px;
`;

const ToolCard = styled.div`
  background: ${({ theme }) => theme.colors.bgSecondary};
  border: 1px solid ${({ theme }) => theme.colors.borderColor};
  border-radius: 12px;
  padding: 24px;
  display: flex;
  flex-direction: column;
  gap: 12px;
  transition: all 0.2s;

  &:hover {
    border-color: ${({ theme }) => theme.colors.accentPrimary};
    transform: translateY(-2px);
  }

  h3 {
    font-size: 18px;
    font-weight: 600;
    color: ${({ theme }) => theme.colors.textPrimary};
  }

  p {
    color: ${({ theme }) => theme.colors.textSecondary};
    font-size: 14px;
    line-height: 1.5;
    flex: 1;
  }
`;

const ToolButton = styled.button`
  padding: 10px 20px;
  background: ${({ theme }) => theme.colors.bgTertiary};
  border: 1px solid ${({ theme }) => theme.colors.borderColor};
  border-radius: 6px;
  color: ${({ theme }) => theme.colors.textPrimary};
  font-size: 14px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s;

  &:hover {
    background: ${({ theme }) => theme.colors.accentPrimary};
    border-color: ${({ theme }) => theme.colors.accentPrimary};
    color: white;
  }
`;

const tools = [
  {
    icon: '🔌',
    title: 'MCP Servers',
    description: 'Connect to Model Context Protocol servers',
    action: 'Configure',
  },
  {
    icon: '⚙️',
    title: 'Custom Tools',
    description: 'Add your own tools and integrations',
    action: 'Manage',
  },
  {
    icon: '📊',
    title: 'Analytics',
    description: 'View usage statistics and insights',
    action: 'View Stats',
  },
  {
    icon: '🔐',
    title: 'API Keys',
    description: 'Manage API keys and credentials',
    action: 'Configure',
  },
];

export const ToolsView: React.FC = () => {
  const handleToolClick = (toolTitle: string) => {
    alert(`${toolTitle}\n\nThis tool configuration will be implemented in a future update.`);
  };

  return (
    <ViewContainer>
      <ViewHeader>
        <h2>Tools & MCP</h2>
      </ViewHeader>
      <ToolsContainer>
        <ToolsGrid>
          {tools.map((tool) => (
            <ToolCard key={tool.title}>
              <h3>
                {tool.icon} {tool.title}
              </h3>
              <p>{tool.description}</p>
              <ToolButton onClick={() => handleToolClick(tool.title)}>
                {tool.action}
              </ToolButton>
            </ToolCard>
          ))}
        </ToolsGrid>
      </ToolsContainer>
    </ViewContainer>
  );
};
