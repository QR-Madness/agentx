/**
 * useSettingsNavigation — Navigation state management for settings sections
 */

import { useState, useCallback } from 'react';

/** Settings open on Overview. (The old default, 'servers', named a section that
 *  stopped existing several reorgs ago — it only ever worked because every call
 *  site passed something else.) */
export function useSettingsNavigation(initialSection = 'overview') {
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
