/**
 * useSettingsNavigation — Navigation state management for settings sections
 */

import { useState, useCallback } from 'react';

export function useSettingsNavigation(initialSection = 'servers') {
  const [activeSection, setActiveSection] = useState(initialSection);
  const [history, setHistory] = useState<string[]>([initialSection]);

  const navigateTo = useCallback((section: string) => {
    setHistory(prev => [...prev, section]);
    setActiveSection(section);
  }, []);

  const goBack = useCallback(() => {
    if (history.length > 1) {
      const newHistory = history.slice(0, -1);
      setHistory(newHistory);
      setActiveSection(newHistory[newHistory.length - 1]);
    }
  }, [history]);

  return {
    activeSection,
    navigateTo,
    goBack,
    canGoBack: history.length > 1
  };
}
