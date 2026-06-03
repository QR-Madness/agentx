import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { CitationElement } from './CitationElement';
import type { CitationElement as CitationElementType } from '../../../lib/exhibits';

const element: CitationElementType = {
  type: 'citation',
  sources: [
    { label: 'NLLB paper', url: 'https://example.com/nllb', quote: 'BLEU +4.2', kind: 'active', source_type: 'web' },
    { label: 'pgvector docs', url: 'https://example.com/pg', kind: 'passive' },
    { label: 'unsafe', url: 'javascript:alert(1)', kind: 'passive' },
  ],
};

describe('CitationElement', () => {
  it('shows an active source quote when folded out', () => {
    render(<CitationElement element={element} messageId="m1" />);
    expect(screen.getByText('NLLB paper')).toBeInTheDocument();
    expect(screen.getByText('BLEU +4.2')).toBeInTheDocument(); // <details> body present in DOM
  });

  it('groups passive sources under a Sources disclosure', () => {
    render(<CitationElement element={element} messageId="m1" />);
    expect(screen.getByText(/Sources \(2\)/)).toBeInTheDocument();
  });

  it('links http(s) urls but renders non-http as inert text', () => {
    render(<CitationElement element={element} messageId="m1" />);
    expect(screen.getByRole('link', { name: 'pgvector docs' })).toHaveAttribute('href', 'https://example.com/pg');
    // javascript: url is not turned into a link
    expect(screen.queryByRole('link', { name: 'unsafe' })).toBeNull();
    expect(screen.getByText('unsafe')).toBeInTheDocument();
  });
});
