/**
 * useSettingsNavigation — which section is showing, and which setting (if any)
 * the user is being sent to.
 *
 * Deep links carry two things: the section to open, and optionally the setting
 * to land on.
 *
 * The focus target carries a sequence number rather than being cleared once
 * consumed. Clearing it on delivery looks tidier but breaks the thing it's for:
 * the state change re-runs the consumer's effect, whose cleanup cancels the
 * scroll that was still in flight, so the flash never lands. The sequence also
 * makes re-selecting the same setting re-flash it, which is what someone
 * clicking the same search hit twice is asking for.
 *
 * Navigating to a section by other means clears the target, so returning to a
 * section later doesn't re-flash something already shown.
 */

import { useState, useCallback, useEffect } from 'react';

export interface SettingFocus {
  /** `store:key` of the setting to land on. */
  key: string;
  /** Distinguishes repeat requests for the same key. */
  seq: number;
}

/** Settings open on Overview. (The old default, 'servers', named a section that
 *  stopped existing several reorgs ago — it only ever worked because every call
 *  site passed something else.) */
export function useSettingsNavigation(
  initialSection = 'overview',
  initialFocus?: string,
) {
  const [activeSection, setActiveSection] = useState(initialSection);
  const [history, setHistory] = useState<string[]>([initialSection]);
  const [focus, setFocus] = useState<SettingFocus | undefined>(
    initialFocus ? { key: initialFocus, seq: 0 } : undefined,
  );

  // A deep link can arrive after mount (the modal is reused), so track the prop.
  useEffect(() => {
    if (!initialFocus) return;
    setFocus(prev => ({ key: initialFocus, seq: (prev?.seq ?? 0) + 1 }));
  }, [initialFocus]);

  const navigateTo = useCallback((section: string) => {
    setHistory(prev => [...prev, section]);
    setActiveSection(section);
    setFocus(undefined);
  }, []);

  /** Go to a section *and* land on one of its settings. */
  const focusSettingIn = useCallback((section: string, settingKey: string) => {
    setHistory(prev => [...prev, section]);
    setActiveSection(section);
    setFocus(prev => ({ key: settingKey, seq: (prev?.seq ?? 0) + 1 }));
  }, []);

  const goBack = useCallback(() => {
    if (history.length > 1) {
      const newHistory = history.slice(0, -1);
      setHistory(newHistory);
      setActiveSection(newHistory[newHistory.length - 1]);
      setFocus(undefined);
    }
  }, [history]);

  return {
    activeSection,
    navigateTo,
    focusSettingIn,
    focus,
    goBack,
    canGoBack: history.length > 1,
  };
}
