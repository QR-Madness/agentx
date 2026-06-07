import { describe, it, expect } from 'vitest';
import { AVATAR_OPTIONS, AVATAR_CATEGORIES, getAvatarIcon, searchAvatars } from './avatars';

describe('avatars catalog', () => {
  it('has unique ids and a valid category for every option', () => {
    const ids = AVATAR_OPTIONS.map(o => o.id);
    expect(new Set(ids).size).toBe(ids.length); // no dupes
    const cats = new Set(AVATAR_CATEGORIES.map(c => c.id));
    for (const o of AVATAR_OPTIONS) expect(cats.has(o.category)).toBe(true);
  });

  it('keeps legacy ids resolvable (saved profiles)', () => {
    for (const id of ['sparkles', 'brain', 'bot', 'radio', 'gem']) {
      expect(getAvatarIcon(id)).toBeTypeOf('object');
      expect(AVATAR_OPTIONS.some(o => o.id === id)).toBe(true);
    }
    expect(getAvatarIcon('does-not-exist')).toBe(getAvatarIcon('sparkles'));
  });

  it('searches over id / label / category / keywords', () => {
    expect(searchAvatars('').length).toBe(AVATAR_OPTIONS.length);
    expect(searchAvatars('rocket').some(o => o.id === 'rocket')).toBe(true);
    expect(searchAvatars('magic').some(o => o.id === 'wand')).toBe(true); // keyword
    expect(searchAvatars('nature').every(o => o.category === 'nature')).toBe(true); // category
    expect(searchAvatars('zzzznope')).toHaveLength(0);
  });
});
