import { describe, it, expect } from 'vitest';
import { composeStack, effectiveContent, estimateTokens, isModified } from './promptStack';
import type { PromptLayer } from './api/types';

function layer(partial: Partial<PromptLayer> & Pick<PromptLayer, 'id' | 'order'>): PromptLayer {
  return {
    id: partial.id,
    title: partial.title ?? partial.id,
    kind: partial.kind ?? 'builtin',
    default: partial.default ?? null,
    default_version: partial.default_version ?? 1,
    override: partial.override ?? null,
    base_version: partial.base_version ?? null,
    effective: '',
    enabled: partial.enabled ?? true,
    order: partial.order,
    modified: partial.modified ?? false,
    update_available: partial.update_available ?? false,
  };
}

describe('composeStack', () => {
  it('joins enabled layers by blank line, in order, override-preferred', () => {
    const layers = [
      layer({ id: 'b', order: 10, default: 'B-default', override: 'B-override' }),
      layer({ id: 'a', order: 0, default: 'A' }),
      layer({ id: 'c', order: 20, default: 'C' }),
    ];
    // Sorted by order: A, B(override), C
    expect(composeStack(layers)).toBe('A\n\nB-override\n\nC');
  });

  it('drops disabled and empty layers (mirrors the backend filter)', () => {
    const layers = [
      layer({ id: 'a', order: 0, default: 'A' }),
      layer({ id: 'off', order: 10, default: 'nope', enabled: false }),
      layer({ id: 'blank', order: 20, default: '   ' }),
      layer({ id: 'c', order: 30, default: 'C' }),
    ];
    expect(composeStack(layers)).toBe('A\n\nC');
  });

  it('uses the default when there is no override', () => {
    expect(effectiveContent({ override: null, default: 'D' })).toBe('D');
    expect(effectiveContent({ override: 'O', default: 'D' })).toBe('O');
    expect(effectiveContent({ override: '', default: 'D' })).toBe(''); // empty override is intentional
  });
});

describe('isModified', () => {
  it('is true only when an override diverges from the default', () => {
    expect(isModified({ override: null, default: 'D' })).toBe(false);
    expect(isModified({ override: 'D', default: 'D' })).toBe(false);
    expect(isModified({ override: 'changed', default: 'D' })).toBe(true);
    expect(isModified({ override: '', default: 'D' })).toBe(true); // cleared text is a real edit
  });
});

describe('estimateTokens', () => {
  it('approximates chars/4 and handles empty', () => {
    expect(estimateTokens('')).toBe(0);
    expect(estimateTokens('abcd')).toBe(1);
    expect(estimateTokens('abcde')).toBe(2);
  });
});
