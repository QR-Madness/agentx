import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { scrollToAnchor } from './scrollToAnchor';

/** rAF is async in jsdom; step past the scheduled attempts. */
const flushFrames = async (n = 3) => {
  for (let i = 0; i < n; i++) await new Promise(r => requestAnimationFrame(() => r(null)));
};

describe('scrollToAnchor', () => {
  beforeEach(() => {
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'] });
    document.body.innerHTML = '';
  });
  afterEach(() => vi.useRealTimers());

  const mount = (value: string) => {
    const el = document.createElement('div');
    el.setAttribute('data-setting', value);
    el.scrollIntoView = vi.fn();
    document.body.appendChild(el);
    return el;
  };

  it('scrolls the target into view and flashes it', async () => {
    const el = mount('memory:recall_candidate_pool');
    scrollToAnchor('setting', 'memory:recall_candidate_pool');
    await flushFrames();

    expect(el.scrollIntoView).toHaveBeenCalledWith({ behavior: 'smooth', block: 'center' });
    expect(el.classList.contains('flash-target')).toBe(true);

    // The mark is temporary — it says "this one", it doesn't stay forever.
    vi.advanceTimersByTime(2000);
    expect(el.classList.contains('flash-target')).toBe(false);
  });

  it('escapes keys containing CSS-significant characters', async () => {
    // Config keys are dotted, and model ids carry colons — both break a naive
    // selector.
    const el = mount('config:context.verbatim_budget_ratio');
    scrollToAnchor('setting', 'config:context.verbatim_budget_ratio');
    await flushFrames();
    expect(el.classList.contains('flash-target')).toBe(true);
  });

  it('waits for content that mounts a few frames late (lazy sections)', async () => {
    scrollToAnchor('setting', 'memory:late', { retries: 5 });
    await flushFrames(2);
    const el = mount('memory:late');
    await flushFrames(4);
    expect(el.classList.contains('flash-target')).toBe(true);
  });

  it('gives up quietly when the target never appears', async () => {
    expect(() => scrollToAnchor('setting', 'memory:ghost', { retries: 2 })).not.toThrow();
    await flushFrames(5);
  });

  it('cleanup removes a pending flash so a timer cannot touch a stale node', async () => {
    const el = mount('memory:x');
    const cancel = scrollToAnchor('setting', 'memory:x');
    await flushFrames();
    expect(el.classList.contains('flash-target')).toBe(true);
    cancel();
    expect(el.classList.contains('flash-target')).toBe(false);
  });

  it('searches only inside the given root', async () => {
    const outside = mount('memory:x');
    const root = document.createElement('div');
    document.body.appendChild(root);

    scrollToAnchor('setting', 'memory:x', { within: root, retries: 1 });
    await flushFrames(3);
    expect(outside.classList.contains('flash-target')).toBe(false);
  });
});
