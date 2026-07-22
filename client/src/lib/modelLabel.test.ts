import { describe, expect, it } from 'vitest';
import { modelShortLabel } from './modelLabel';

describe('modelShortLabel', () => {
  it('strips provider prefix and namespace to the model tail', () => {
    expect(modelShortLabel('openrouter:deepseek/deepseek-v4-flash')).toBe('deepseek-v4-flash');
    expect(modelShortLabel('vercel:nvidia/nemotron-3-ultra-550b')).toBe('nemotron-3-ultra-550b');
    expect(modelShortLabel('openrouter:thinkingmachines/inkling')).toBe('inkling');
  });

  it('passes a bare model id through unchanged', () => {
    expect(modelShortLabel('gpt-4o')).toBe('gpt-4o');
  });

  it('handles a provider prefix with no namespace slash', () => {
    expect(modelShortLabel('lmstudio:local-model')).toBe('local-model');
  });

  it('returns null for empty/unset so callers can show "inherits default"', () => {
    expect(modelShortLabel(undefined)).toBeNull();
    expect(modelShortLabel(null)).toBeNull();
    expect(modelShortLabel('')).toBeNull();
  });
});
