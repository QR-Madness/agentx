import { describe, it, expect } from 'vitest';
import { buildResumeNudge } from './planResume';
import type { PlanStatusResponse } from './api/types';

const status = (subtasks: PlanStatusResponse['subtasks']): PlanStatusResponse => ({
  found: true,
  plan_id: 'p1',
  session_id: 's1',
  subtasks,
});

describe('buildResumeNudge', () => {
  it('carries each completed step result so prior context is re-attached', () => {
    const msg = buildResumeNudge(status([
      { id: 0, status: 'complete', description: 'Search for sources', result: 'Found 3 strong sources on the topic.' },
      { id: 1, status: 'pending', description: 'Synthesize findings' },
    ]));
    // The completed step's result is present (not just its description).
    expect(msg).toContain('Found 3 strong sources on the topic.');
    expect(msg).toContain('Search for sources');
    // The remaining step is listed as still-to-do.
    expect(msg).toContain('Synthesize findings');
    expect(msg).toContain('continue from where we left off');
  });

  it('truncates an oversized step result but keeps the head', () => {
    const big = 'x'.repeat(5000);
    const msg = buildResumeNudge(status([
      { id: 0, status: 'complete', description: 'Big step', result: big },
    ]));
    expect(msg).toContain('[truncated]');
    expect(msg).toContain('Big step');
    expect(msg.length).toBeLessThan(big.length);
  });

  it('falls back to the description when a completed step has no result', () => {
    const msg = buildResumeNudge(status([
      { id: 0, status: 'complete', description: 'Did a thing' },
    ]));
    expect(msg).toContain('Did a thing');
    expect(msg).not.toContain('[truncated]');
  });
});
