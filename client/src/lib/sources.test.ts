import { describe, it, expect } from 'vitest';
import { Globe, Database, BookMarked, Link2 } from 'lucide-react';
import { hostTeaser, dominantIcon, sourceIcon, sourceText } from './sources';
import type { CitationSource } from './exhibits';

const web = (host: string): CitationSource => ({
  label: host,
  url: `https://${host}/x`,
  kind: 'passive',
  source_type: 'web',
});

describe('hostTeaser', () => {
  it('names the first two distinct hosts and folds the rest into +N', () => {
    expect(hostTeaser([web('a.com'), web('b.com'), web('c.com'), web('d.com')])).toBe(
      'a.com, b.com +2',
    );
  });

  it('dedupes repeated hosts before counting', () => {
    expect(hostTeaser([web('a.com'), web('a.com'), web('b.com')])).toBe('a.com, b.com');
  });

  it('falls back to the source-type word for non-http sources', () => {
    expect(hostTeaser([{ label: 'note', kind: 'passive', source_type: 'memory' }])).toBe('memory');
  });

  it('is empty for no sources', () => {
    expect(hostTeaser([])).toBe('');
  });
});

describe('dominantIcon', () => {
  it('uses the type icon when every source shares a type', () => {
    expect(dominantIcon([web('a.com'), web('b.com')])).toBe(Globe);
  });

  it('falls back to the book icon for mixed types', () => {
    expect(
      dominantIcon([web('a.com'), { label: 'm', kind: 'passive', source_type: 'memory' }]),
    ).toBe(BookMarked);
  });
});

describe('sourceIcon / sourceText', () => {
  it('maps type to icon, defaulting to a generic link', () => {
    expect(sourceIcon({ source_type: 'memory' })).toBe(Database);
    expect(sourceIcon({ source_type: undefined })).toBe(Link2);
  });

  it('prefers the label, falling back to host then url', () => {
    expect(sourceText({ label: 'Title', url: 'https://a.com', kind: 'passive' })).toBe('Title');
    expect(sourceText({ label: '', url: 'https://a.com/p', kind: 'passive' })).toBe('a.com');
  });
});
