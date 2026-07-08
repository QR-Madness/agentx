import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { CitationElement } from './CitationElement';
import type { CitationElement as CitationElementType } from '../../../lib/exhibits';

const mixed: CitationElementType = {
  type: 'citation',
  sources: [
    { label: 'NLLB paper', url: 'https://example.com/nllb', quote: 'BLEU +4.2', kind: 'active', source_type: 'web' },
    { label: 'pgvector docs', url: 'https://other.com/pg', kind: 'passive' },
    { label: 'unsafe', url: 'javascript:alert(1)', kind: 'passive' },
  ],
};

/** Expand the collapsed-by-default header so the source list is in the DOM. */
async function expand(user: ReturnType<typeof userEvent.setup>) {
  await user.click(screen.getByRole('button', { name: /Sources/ }));
}

describe('CitationElement', () => {
  it('always shows a Sources header with the total count', () => {
    render(<CitationElement element={mixed} messageId="m1" />);
    expect(screen.getByText(/^Sources$/)).toBeInTheDocument();
    expect(screen.getByText('· 3')).toBeInTheDocument();
  });

  it('is collapsed by default and toggles open on the header', async () => {
    const user = userEvent.setup();
    render(<CitationElement element={mixed} messageId="m1" />);
    const header = screen.getByRole('button', { name: /Sources/ });
    expect(header).toHaveAttribute('aria-expanded', 'false');
    // Body content absent while collapsed.
    expect(screen.queryByText('NLLB paper')).toBeNull();
    await user.click(header);
    expect(header).toHaveAttribute('aria-expanded', 'true');
    expect(screen.getByText('NLLB paper')).toBeInTheDocument();
  });

  it('shows an active source quote when expanded and folded out', async () => {
    const user = userEvent.setup();
    render(<CitationElement element={mixed} messageId="m1" />);
    await expand(user);
    expect(screen.getByText('NLLB paper')).toBeInTheDocument();
    expect(screen.getByText('BLEU +4.2')).toBeInTheDocument(); // <details> body present in DOM
  });

  it('tucks passive sources under a "More sources" disclosure when active exist', async () => {
    const user = userEvent.setup();
    render(<CitationElement element={mixed} messageId="m1" />);
    await expand(user);
    expect(screen.getByText(/More sources \(2\)/)).toBeInTheDocument();
  });

  it('links http(s) urls but renders non-http as inert text', async () => {
    const user = userEvent.setup();
    render(<CitationElement element={mixed} messageId="m1" />);
    await expand(user);
    expect(screen.getByRole('link', { name: 'pgvector docs' })).toHaveAttribute(
      'href',
      'https://other.com/pg',
    );
    expect(screen.queryByRole('link', { name: 'unsafe' })).toBeNull();
    expect(screen.getByText('unsafe')).toBeInTheDocument();
  });

  it('renders a passive-only (auto-captured) list open once expanded, capped with "+N more"', async () => {
    const user = userEvent.setup();
    const passiveOnly: CitationElementType = {
      type: 'citation',
      sources: Array.from({ length: 8 }, (_, i) => ({
        label: `Result ${i}`,
        url: `https://example.com/${i}`,
        kind: 'passive' as const,
        source_type: 'web' as const,
      })),
    };
    render(<CitationElement element={passiveOnly} messageId="m1" />);
    await expand(user);
    // No inner disclosure to open — the first 5 are visible immediately.
    expect(screen.getByRole('link', { name: 'Result 0' })).toBeInTheDocument();
    expect(screen.queryByRole('link', { name: 'Result 7' })).toBeNull();
    await user.click(screen.getByRole('button', { name: '+3 more' }));
    expect(screen.getByRole('link', { name: 'Result 7' })).toBeInTheDocument();
  });
});
