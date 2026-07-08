import { describe, it, expect } from 'vitest';
import { parseNofM, compactNumber } from './progress';
import { nextMessage, CONSOLIDATION_MESSAGES } from './consolidation-messages';

describe('parseNofM', () => {
  it('parses "N of M" to a clamped fraction', () => {
    expect(parseNofM('3 of 10')).toBeCloseTo(0.3);
    expect(parseNofM('1 of 1')).toBe(1);
    expect(parseNofM('10 of 10')).toBe(1);
  });
  it('returns null for junk / zero total / non-strings', () => {
    expect(parseNofM('processing')).toBeNull();
    expect(parseNofM('5 of 0')).toBeNull();
    expect(parseNofM(undefined)).toBeNull();
    expect(parseNofM(42)).toBeNull();
  });
});

describe('compactNumber', () => {
  it('formats magnitudes', () => {
    expect(compactNumber(0)).toBe('0');
    expect(compactNumber(999)).toBe('999');
    expect(compactNumber(1200)).toBe('1.2K');
    expect(compactNumber(45300)).toBe('45K');
    expect(compactNumber(3_400_000)).toBe('3.4M');
  });
});

describe('nextMessage', () => {
  it('never returns the previous message', () => {
    let prev = CONSOLIDATION_MESSAGES[0];
    for (let i = 0; i < 50; i++) {
      const next = nextMessage(prev);
      expect(next).not.toBe(prev);
      expect(CONSOLIDATION_MESSAGES).toContain(next);
      prev = next;
    }
  });
});
