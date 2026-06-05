import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { CitationElement } from './CitationElement';
import type { CitationElement as CitationElementType } from '../../../lib/exhibits';

const mixed: CitationElementType = {
  type: 'citation',
  sources: [
    { label: 'NLLB paper', url: 'https://example.com/nllb', quote: 'BLEU +4.2', kind: 'active', source_type: 'web' },
    { label: 'pgvector docs', url: 'https://example.com/pg', kind: 'passive' },
    { label: 'unsafe', url: 'javascript:alert(1)', kind: 'passive' },
  ],
};

describe('CitationElement', () => {
  it('always shows a Sources header with the total count', () => {
    render(<CitationElement element={mixed} messageId="m1" />);
    expect(screen.getByText(/^Sources$/)).toBeInTheDocument();
    expect(screen.getByText('· 3')).toBeInTheDocument();
  });

  it('shows an active source quote when folded out', () => {
    render(<CitationElement element={mixed} messageId="m1" />);
    expect(screen.getByText('NLLB paper')).toBeInTheDocument();
    expect(screen.getByText('BLEU +4.2')).toBeInTheDocument(); // <details> body present in DOM
  });

  it('tucks passive sources under a "More sources" disclosure when active exist', () => {
    render(<CitationElement element={mixed} messageId="m1" />);
    expect(screen.getByText(/More sources \(2\)/)).toBeInTheDocument();
  });

  it('links http(s) urls but renders non-http as inert text', () => {
    render(<CitationElement element={mixed} messageId="m1" />);
    expect(screen.getByRole('link', { name: 'pgvector docs' })).toHaveAttribute(
      'href',
      'https://example.com/pg',
    );
    expect(screen.queryByRole('link', { name: 'unsafe' })).toBeNull();
    expect(screen.getByText('unsafe')).toBeInTheDocument();
  });

  it('renders a passive-only (auto-captured) list open, capped with "+N more"', async () => {
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
    // No disclosure to open — the first 5 are visible immediately.
    expect(screen.getByRole('link', { name: 'Result 0' })).toBeInTheDocument();
    expect(screen.queryByRole('link', { name: 'Result 7' })).toBeNull();
    await user.click(screen.getByRole('button', { name: '+3 more' }));
    expect(screen.getByRole('link', { name: 'Result 7' })).toBeInTheDocument();
  });
});
