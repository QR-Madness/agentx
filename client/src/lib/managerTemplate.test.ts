import { describe, expect, it } from 'vitest';
import { MANAGER_BLOCKED_TOOLS, mergeManagerTemplate } from './managerTemplate';

describe('managerTemplate', () => {
  it('pins the mirrored blocklist (must match api MANAGER_REPORT_ONLY_BLOCKED_TOOLS)', () => {
    // Drift guard: if this fails, someone changed one side of the mirror —
    // update BOTH this list and api/agentx_ai/agent/models.py together.
    expect(MANAGER_BLOCKED_TOOLS).toEqual([
      '_internal.create_document',
      '_internal.update_document',
      '_internal.append_to_document',
      '_internal.edit_document',
      '_internal.rename_document',
      '_internal.delete_document',
      '_internal.run_command',
      '_internal.write_file',
      '_internal.generate_image',
      '_internal.generate_speech',
    ]);
  });

  it('merges as an order-preserving union (idempotent, user entries first)', () => {
    const merged = mergeManagerTemplate(['custom.tool', '_internal.run_command']);
    expect(merged[0]).toBe('custom.tool');
    expect(merged.filter(t => t === '_internal.run_command')).toHaveLength(1);
    for (const t of MANAGER_BLOCKED_TOOLS) expect(merged).toContain(t);
    // Re-applying changes nothing (client+server double-merge converges).
    expect(mergeManagerTemplate(merged)).toEqual(merged);
  });

  it('handles empty input', () => {
    expect(mergeManagerTemplate(undefined)).toEqual([...MANAGER_BLOCKED_TOOLS]);
    expect(mergeManagerTemplate(null)).toEqual([...MANAGER_BLOCKED_TOOLS]);
  });
});
