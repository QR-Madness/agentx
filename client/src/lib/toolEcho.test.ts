/**
 * Tool-name echo hygiene — the poisoned-transcript regression.
 *
 * Some routes leak the called function's name into the content stream right
 * before the structured tool call ("use_skillGot it — …"). The server strips
 * the echo from the stored turn (tool_loop._strip_tool_name_echo); this is the
 * client twin applied at the tool_call flush seam. The boundary rule must
 * match the server's: strip only at position 0, only when the remainder is
 * empty or starts non-whitespace.
 */

import { describe, it, expect } from 'vitest';
import { stripToolNameEcho } from './messages';

describe('stripToolNameEcho', () => {
  it('strips a glued echo at position 0', () => {
    expect(stripToolNameEcho('use_skillGot it — done.', 'use_skill'))
      .toBe('Got it — done.');
  });

  it('strips a bare-name-only buffer to empty', () => {
    expect(stripToolNameEcho('use_skill', 'use_skill')).toBe('');
  });

  it('strips repeated echoes of the same name', () => {
    expect(stripToolNameEcho('use_skilluse_skillOn it.', 'use_skill'))
      .toBe('On it.');
  });

  it('keeps prose that talks ABOUT the tool (space boundary)', () => {
    expect(stripToolNameEcho('use_skill returned nothing.', 'use_skill'))
      .toBe('use_skill returned nothing.');
  });

  it('keeps a mid-text occurrence', () => {
    expect(stripToolNameEcho('Calling use_skill now.', 'use_skill'))
      .toBe('Calling use_skill now.');
  });

  it('tolerates leading whitespace before the echo', () => {
    expect(stripToolNameEcho('\n\nuse_skillHere we go', 'use_skill'))
      .toBe('\n\nHere we go');
  });

  it('is a no-op for an empty tool name', () => {
    expect(stripToolNameEcho('anything', '')).toBe('anything');
  });
});
