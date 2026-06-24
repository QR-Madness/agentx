import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, waitFor } from '@testing-library/react';
import { AgentAvatar } from './AgentAvatar';

// Resolve a media: avatar to a deterministic object URL per ref, so we can assert the
// component never shows a *stale* image and never gets stuck on the icon fallback.
vi.mock('../../lib/mediaImage', () => ({
  resolveAvatarImage: vi.fn(async (avatar: string) => `blob:${avatar}`),
}));

import { resolveAvatarImage } from '../../lib/mediaImage';

describe('AgentAvatar', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders an <img> for a media avatar, resolved to its own URL', async () => {
    const { container } = render(<AgentAvatar avatar="media:ws_home/doc_A" />);
    await waitFor(() => {
      const img = container.querySelector('img');
      expect(img).not.toBeNull();
      expect(img?.getAttribute('src')).toBe('blob:media:ws_home/doc_A');
    });
  });

  it('never shows the previous agent’s image after the avatar prop changes', async () => {
    const { container, rerender } = render(<AgentAvatar avatar="media:ws_home/doc_A" />);
    await waitFor(() => expect(container.querySelector('img')?.getAttribute('src')).toBe('blob:media:ws_home/doc_A'));

    // Switch to a different agent. The stale URL must never be shown for the new avatar:
    // until the new one resolves, fall back to the icon (no <img> with the old src).
    rerender(<AgentAvatar avatar="media:ws_home/doc_B" />);
    expect(container.querySelector('img')?.getAttribute('src')).not.toBe('blob:media:ws_home/doc_A');

    await waitFor(() => expect(container.querySelector('img')?.getAttribute('src')).toBe('blob:media:ws_home/doc_B'));
  });

  it('falls back to the lucide icon for a non-media (icon id) avatar', () => {
    const { container } = render(<AgentAvatar avatar="brain" />);
    expect(container.querySelector('img')).toBeNull();
    expect(container.querySelector('svg')).not.toBeNull();
    expect(resolveAvatarImage).not.toHaveBeenCalled();
  });

  it('renders the icon for an undefined avatar', () => {
    const { container } = render(<AgentAvatar avatar={undefined} />);
    expect(container.querySelector('img')).toBeNull();
    expect(container.querySelector('svg')).not.toBeNull();
  });
});
