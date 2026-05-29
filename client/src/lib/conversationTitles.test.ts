import { describe, it, expect, beforeEach } from 'vitest';
import {
  getTitleOverride,
  setTitleOverride,
  clearTitleOverride,
  getDisplayTitle,
} from './conversationTitles';

// getActiveServerId reads this key (see storage.ts STORAGE_KEYS.activeServer).
const ACTIVE_SERVER_KEY = 'agentx:activeServer';

describe('conversationTitles', () => {
  beforeEach(() => {
    localStorage.clear();
    localStorage.setItem(ACTIVE_SERVER_KEY, 'srv1');
  });

  it('returns undefined / falls back when no override is set', () => {
    expect(getTitleOverride('c1')).toBeUndefined();
    expect(getDisplayTitle('c1', 'Server Title')).toBe('Server Title');
  });

  it('sets and reads an override', () => {
    setTitleOverride('c1', 'My Renamed Chat');
    expect(getTitleOverride('c1')).toBe('My Renamed Chat');
    expect(getDisplayTitle('c1', 'Server Title')).toBe('My Renamed Chat');
  });

  it('trims and treats a blank title as a clear', () => {
    setTitleOverride('c1', 'X');
    setTitleOverride('c1', '   ');
    expect(getTitleOverride('c1')).toBeUndefined();
  });

  it('clearTitleOverride reverts to the fallback', () => {
    setTitleOverride('c1', 'X');
    clearTitleOverride('c1');
    expect(getDisplayTitle('c1', 'fallback')).toBe('fallback');
  });

  it('scopes overrides per server', () => {
    setTitleOverride('c1', 'Server1 name');
    localStorage.setItem(ACTIVE_SERVER_KEY, 'srv2');
    expect(getTitleOverride('c1')).toBeUndefined();
    localStorage.setItem(ACTIVE_SERVER_KEY, 'srv1');
    expect(getTitleOverride('c1')).toBe('Server1 name');
  });

  it('returns the fallback for an empty conversation id', () => {
    expect(getDisplayTitle(null, 'fallback')).toBe('fallback');
    expect(getDisplayTitle(undefined, 'fallback')).toBe('fallback');
  });
});
