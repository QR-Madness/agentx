import React from 'react';
import styled from 'styled-components';
import { useTheme } from '../theme/ThemeContext';

const SidebarContainer = styled.aside`
  width: ${({ theme }) => theme.sidebar.width};
  background: ${({ theme }) => theme.colors.bgSecondary};
  border-right: 1px solid ${({ theme }) => theme.colors.borderColor};
  display: flex;
  flex-direction: column;
  padding: 20px;
`;

const SidebarHeader = styled.div`
  margin-bottom: 32px;
`;

const AppTitle = styled.h1`
  font-size: 28px;
  font-weight: 700;
  color: ${({ theme }) => theme.colors.textPrimary};
  margin-bottom: 4px;
`;

const AppSubtitle = styled.p`
  font-size: 13px;
  color: ${({ theme }) => theme.colors.textMuted};
  font-weight: 400;
`;

const NavMenu = styled.nav`
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 8px;
`;

const NavItem = styled.button<{ $active: boolean }>`
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 12px 16px;
  background: ${({ $active, theme }) =>
    $active ? theme.colors.accentPrimary : 'transparent'};
  border: none;
  border-radius: 8px;
  color: ${({ $active, theme }) =>
    $active ? 'white' : theme.colors.textSecondary};
  cursor: pointer;
  font-size: 15px;
  transition: all 0.2s;
  text-align: left;

  &:hover {
    background: ${({ $active, theme }) =>
      $active ? theme.colors.accentPrimary : theme.colors.bgHover};
    color: ${({ $active, theme }) =>
      $active ? 'white' : theme.colors.textPrimary};
  }
`;

const NavIcon = styled.span`
  font-size: 20px;
`;

const NavLabel = styled.span`
  font-weight: 500;
`;

const SidebarFooter = styled.div`
  margin-top: 16px;
`;

const ThemeToggle = styled.button`
  display: flex;
  align-items: center;
  gap: 12px;
  width: 100%;
  padding: 12px 16px;
  background: ${({ theme }) => theme.colors.bgTertiary};
  border: 1px solid ${({ theme }) => theme.colors.borderColor};
  border-radius: 8px;
  color: ${({ theme }) => theme.colors.textSecondary};
  cursor: pointer;
  font-size: 15px;
  transition: all 0.2s;

  &:hover {
    background: ${({ theme }) => theme.colors.bgHover};
    color: ${({ theme }) => theme.colors.textPrimary};
  }
`;

const ThemeIcon = styled.span`
  font-size: 18px;
`;

const ThemeLabel = styled.span`
  font-weight: 500;
`;

interface SidebarProps {
  activeView: string;
  onViewChange: (view: string) => void;
}

const navItems = [
  { id: 'chat', icon: '💬', label: 'Chat' },
  { id: 'files', icon: '📁', label: 'File Analysis' },
  { id: 'translate', icon: '🌐', label: 'Translation' },
  { id: 'tools', icon: '🔧', label: 'Tools/MCP' },
];

export const Sidebar: React.FC<SidebarProps> = ({ activeView, onViewChange }) => {
  const { cycleTheme, themeName } = useTheme();

  const themeOrder = ['dark', 'light', 'ocean', 'forest'];
  const currentIndex = themeOrder.indexOf(themeName);
  const nextTheme = themeOrder[(currentIndex + 1) % themeOrder.length];
  const nextThemeCapitalized = nextTheme.charAt(0).toUpperCase() + nextTheme.slice(1);

  return (
    <SidebarContainer>
      <SidebarHeader>
        <AppTitle>AgentX</AppTitle>
        <AppSubtitle>AI Toolbox</AppSubtitle>
      </SidebarHeader>

      <NavMenu>
        {navItems.map((item) => (
          <NavItem
            key={item.id}
            $active={activeView === item.id}
            onClick={() => onViewChange(item.id)}
          >
            <NavIcon>{item.icon}</NavIcon>
            <NavLabel>{item.label}</NavLabel>
          </NavItem>
        ))}
      </NavMenu>

      <SidebarFooter>
        <ThemeToggle onClick={cycleTheme}>
          <ThemeIcon>🎨</ThemeIcon>
          <ThemeLabel>Theme: {nextThemeCapitalized}</ThemeLabel>
        </ThemeToggle>
      </SidebarFooter>
    </SidebarContainer>
  );
};
