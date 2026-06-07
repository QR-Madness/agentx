import { describe, it, expect, beforeEach } from 'vitest';
import { setActiveServerId } from './storage';
import {
  getMeta, patchMeta, clearMeta, listGroups,
  getDisplayTitle, setTitleOverride, clearTitleOverride,
  __resetMetaCacheForTests,
} from './conversationMeta';

describe('conversationMeta', () => {
  beforeEach(() => {
    localStorage.clear();
    __resetMetaCacheForTests();
    setActiveServerId('srv1');
  });

  it('returns empty meta by default', () => {
    expect(getMeta('c1')).toEqual({});
  });

  it('patches and reads back fields', () => {
    patchMeta('c1', { pinned: true, color: 'blue' });
    expect(getMeta('c1')).toEqual({ pinned: true, color: 'blue' });
  });

  it('prunes falsy values (unpin clears the key)', () => {
    patchMeta('c1', { pinned: true });
    patchMeta('c1', { pinned: false });
    expect(getMeta('c1').pinned).toBeUndefined();
  });

  it('clears meta entirely', () => {
    patchMeta('c1', { archived: true });
    clearMeta('c1');
    expect(getMeta('c1')).toEqual({});
  });

  it('lists distinct groups, sorted', () => {
    patchMeta('a', { group: 'Work' });
    patchMeta('b', { group: 'Personal' });
    patchMeta('c', { group: 'Work' });
    expect(listGroups()).toEqual(['Personal', 'Work']);
  });

  it('title shim reads/writes meta.title', () => {
    expect(getDisplayTitle('c1', 'Fallback')).toBe('Fallback');
    setTitleOverride('c1', 'Custom');
    expect(getDisplayTitle('c1', 'Fallback')).toBe('Custom');
    clearTitleOverride('c1');
    expect(getDisplayTitle('c1', 'Fallback')).toBe('Fallback');
  });

  it('is scoped per server', () => {
    patchMeta('c1', { pinned: true });
    setActiveServerId('srv2');
    expect(getMeta('c1')).toEqual({});
    setActiveServerId('srv1');
    expect(getMeta('c1').pinned).toBe(true);
  });

  it('migrates legacy convTitles into meta.title once', () => {
    setActiveServerId('srvLegacy');
    localStorage.setItem('agentx:server:srvLegacy:convTitles', JSON.stringify({ old1: 'Legacy Title' }));
    expect(getDisplayTitle('old1', 'fallback')).toBe('Legacy Title');
  });
});
