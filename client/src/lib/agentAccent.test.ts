import { describe, it, expect } from 'vitest';
import { agentAccent } from './agentAccent';

describe('agentAccent', () => {
  it('is deterministic for a given id', () => {
    expect(agentAccent('bright-grand-fern')).toEqual(agentAccent('bright-grand-fern'));
    expect(agentAccent('bright-grand-fern').hue).toBe(agentAccent('bright-grand-fern').hue);
  });

  it('varies by id and stays in range', () => {
    const a = agentAccent('mobius-aa-bb');
    const b = agentAccent('jeff-cc-dd');
    expect(a.hue).not.toBe(b.hue); // (collisions possible in general, but not for these)
    for (const id of ['', 'x', 'a-long-agent-id-token', 'Z']) {
      const { hue } = agentAccent(id);
      expect(hue).toBeGreaterThanOrEqual(0);
      expect(hue).toBeLessThan(360);
    }
  });

  it('falls back gracefully for empty/nullish ids', () => {
    expect(agentAccent(undefined).accent).toMatch(/^hsl\(/);
    expect(agentAccent(null).soft).toContain('/ 0.14');
  });
});
