import { describe, it, expect } from 'vitest';
import { formatTimestamp } from './formatTimestamp';

describe('formatTimestamp', () => {
  it('returns "Never" for undefined', () => {
    expect(formatTimestamp(undefined)).toBe('Never');
  });

  it('returns "Just now" for the current moment', () => {
    expect(formatTimestamp(new Date().toISOString())).toBe('Just now');
  });

  it('formats minutes ago', () => {
    const tenMinAgo = new Date(Date.now() - 10 * 60_000).toISOString();
    expect(formatTimestamp(tenMinAgo)).toBe('10m ago');
  });

  it('formats hours ago', () => {
    const threeHoursAgo = new Date(Date.now() - 3 * 3_600_000).toISOString();
    expect(formatTimestamp(threeHoursAgo)).toBe('3h ago');
  });

  it('formats days ago', () => {
    const twoDaysAgo = new Date(Date.now() - 2 * 86_400_000).toISOString();
    expect(formatTimestamp(twoDaysAgo)).toBe('2d ago');
  });

  it('falls back to a locale date beyond a week', () => {
    const old = new Date(Date.now() - 30 * 86_400_000).toISOString();
    const out = formatTimestamp(old);
    expect(out).not.toMatch(/ago|Just now|Never/);
  });
});
