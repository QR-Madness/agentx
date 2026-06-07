import { describe, it, expect, beforeEach } from 'vitest';
import { getRecentCommandIds, pushRecentCommand } from './recentCommands';

describe('recentCommands', () => {
  beforeEach(() => localStorage.clear());

  it('returns empty by default', () => {
    expect(getRecentCommandIds()).toEqual([]);
  });

  it('pushes most-recent-first', () => {
    pushRecentCommand('a');
    pushRecentCommand('b');
    expect(getRecentCommandIds()).toEqual(['b', 'a']);
  });

  it('dedupes, promoting the re-used id to the front', () => {
    pushRecentCommand('a');
    pushRecentCommand('b');
    pushRecentCommand('a');
    expect(getRecentCommandIds()).toEqual(['a', 'b']);
  });

  it('caps the list length', () => {
    for (let i = 0; i < 20; i++) pushRecentCommand(`cmd-${i}`);
    const ids = getRecentCommandIds();
    expect(ids.length).toBeLessThanOrEqual(6);
    expect(ids[0]).toBe('cmd-19'); // newest first
  });
});
