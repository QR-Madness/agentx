import React, {useState} from 'react';

import styled from 'styled-components';
import {Sidebar} from './components/Sidebar';
import {ChatView} from './components/views/ChatView';
import {FileAnalysisView} from './components/views/FileAnalysisView';
import {TranslationView} from './components/views/TranslationView';
import {ToolsView} from './components/views/ToolsView';
import {ThemeProvider} from "./theme/ThemeContext";

const AppContainer = styled.div`
    display: flex;
    height: 100%;
    width: 100%;
`;

const MainContent = styled.main`
    flex: 1;
    background: ${({theme}) => theme.colors.bgPrimary};
    overflow: hidden;
`;

export const AppPage: () => any = () => {
  const [activeView, setActiveView] = useState('chat');

  const renderView = () => {
    switch (activeView) {
      case 'chat':
        return <ChatView/>;
      case 'files':
        return <FileAnalysisView/>;
      case 'translate':
        return <TranslationView/>;
      case 'tools':
        return <ToolsView/>;
      default:
        return <ChatView/>;
    }
  };

  console.log('AgentX initialized successfully!');

  return <ThemeProvider>
    <AppContainer>
      <Sidebar activeView={activeView} onViewChange={setActiveView}/>
      <MainContent>{renderView()}</MainContent>
    </AppContainer>
  </ThemeProvider>
};

