import { describe, it, expect } from 'vitest';
import { isNumericColumn, sortRows } from './tableSort';

const rows = [
  ['opus', '0.40', '1200'],
  ['haiku', '0.05', '900'],
  ['sonnet', '', '1050'],
];

describe('isNumericColumn', () => {
  it('detects a numeric column (ignoring blanks, $ and %)', () => {
    expect(isNumericColumn([['$1.2'], ['3'], ['']], 0)).toBe(true);
    expect(isNumericColumn(rows, 1)).toBe(true);
    expect(isNumericColumn(rows, 2)).toBe(true);
  });
  it('treats a text column as non-numeric', () => {
    expect(isNumericColumn(rows, 0)).toBe(false);
  });
  it('is non-numeric when there are no values', () => {
    expect(isNumericColumn([[''], ['']], 0)).toBe(false);
  });
});

describe('sortRows', () => {
  it('sorts a numeric column numerically (not lexically)', () => {
    const out = sortRows(rows, 2, 'asc').map((r) => r[2]);
    expect(out).toEqual(['900', '1050', '1200']); // numeric, not '1050' < '1200' < '900'
  });
  it('sorts a text column lexically and respects direction', () => {
    expect(sortRows(rows, 0, 'asc').map((r) => r[0])).toEqual(['haiku', 'opus', 'sonnet']);
    expect(sortRows(rows, 0, 'desc').map((r) => r[0])).toEqual(['sonnet', 'opus', 'haiku']);
  });
  it('sinks blank cells to the bottom regardless of direction', () => {
    const last = (dir: 'asc' | 'desc') => {
      const labels = sortRows(rows, 1, dir).map((r) => r[0]);
      return labels[labels.length - 1];
    };
    expect(last('asc')).toBe('sonnet');
    expect(last('desc')).toBe('sonnet');
  });
});
