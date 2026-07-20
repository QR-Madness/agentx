import { describe, expect, it } from 'vitest';
import { deserializeWorkflow, serializeWorkflow } from './alloy';
import type { ServerWorkflow } from './types';

const serverWf: ServerWorkflow = {
  id: 't1',
  name: 'Team One',
  description: null,
  supervisor_agent_id: 'lead-aa',
  manager_agent_id: 'mgr-bb',
  members: [
    { agent_id: 'lead-aa', role: 'supervisor' },
    { agent_id: 'spec-cc', role: 'specialist', delegation_hint: 'analysis' },
  ],
  routes: [],
  shared_channel: '_alloy_t1',
  canvas: {},
  created_at: null,
  updated_at: null,
};

describe('alloy workflow serializers — org fields', () => {
  it('deserializes manager_agent_id (null when absent — org-free team)', () => {
    expect(deserializeWorkflow(serverWf).managerAgentId).toBe('mgr-bb');
    const { manager_agent_id: _drop, ...legacy } = serverWf;
    expect(deserializeWorkflow(legacy as ServerWorkflow).managerAgentId).toBeNull();
  });

  it('serializes managerAgentId, passing null through to clear (PATCH)', () => {
    const wf = deserializeWorkflow(serverWf);
    expect(serializeWorkflow({ ...wf, routes: undefined })).toMatchObject({
      supervisor_agent_id: 'lead-aa',
      manager_agent_id: 'mgr-bb',
    });
    expect(serializeWorkflow({ managerAgentId: null })).toEqual({ manager_agent_id: null });
    // Omitted field stays omitted (PATCH must not touch it).
    expect(serializeWorkflow({ name: 'x' })).not.toHaveProperty('manager_agent_id');
  });
});
