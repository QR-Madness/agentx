/**
 * UIChromeContext — app-chrome visibility state (Focus / Zen mode).
 *
 * Focus mode strips the titlebar down to the essentials and hides the chat
 * header so the conversation canvas fills the window. Toggled from the strip
 * and from the command palette (⌘K).
 */

import { createContext, useCallback, useContext, useMemo, useState, type ReactNode } from 'react';

interface UIChromeContextValue {
  focusMode: boolean;
  toggleFocusMode: () => void;
  setFocusMode: (on: boolean) => void;
}

const UIChromeContext = createContext<UIChromeContextValue | undefined>(undefined);

export function UIChromeProvider({ children }: { children: ReactNode }) {
  const [focusMode, setFocusMode] = useState(false);

  const toggleFocusMode = useCallback(() => setFocusMode(prev => !prev), []);

  const value = useMemo(
    () => ({ focusMode, toggleFocusMode, setFocusMode }),
    [focusMode, toggleFocusMode],
  );

  return <UIChromeContext.Provider value={value}>{children}</UIChromeContext.Provider>;
}

export function useUIChrome(): UIChromeContextValue {
  const ctx = useContext(UIChromeContext);
  if (!ctx) throw new Error('useUIChrome must be used within UIChromeProvider');
  return ctx;
}
