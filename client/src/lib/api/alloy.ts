import { request as apiRequest } from './core';
import type { AlloyWorkflow, AlloyWorkflowCreate, AlloyWorkflowUpdate, ServerWorkflow } from './types';

// Map the server's snake_case workflow shape to/from the client camelCase model.
function deserializeWorkflow(w: ServerWorkflow): AlloyWorkflow {
  return {
    id: w.id,
    name: w.name,
    description: w.description ?? undefined,
    supervisorAgentId: w.supervisor_agent_id,
    members: w.members.map((m) => ({
      agentId: m.agent_id,
      role: m.role,
      delegationHint: m.delegation_hint,
    })),
    routes: (w.routes || []).map((r) => ({
      fromAgentId: r.from_agent_id,
      toAgentId: r.to_agent_id,
      when: r.when,
    })),
    sharedChannel: w.shared_channel,
    canvas: w.canvas || {},
    createdAt: w.created_at,
    updatedAt: w.updated_at,
  };
}

function serializeWorkflow(
  w: AlloyWorkflowCreate | AlloyWorkflowUpdate
): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  if ('id' in w && w.id !== undefined) out.id = w.id;
  if (w.name !== undefined) out.name = w.name;
  if (w.description !== undefined) out.description = w.description;
  if (w.supervisorAgentId !== undefined) out.supervisor_agent_id = w.supervisorAgentId;
  if (w.members !== undefined) {
    out.members = w.members.map((m) => ({
      agent_id: m.agentId,
      role: m.role,
      ...(m.delegationHint ? { delegation_hint: m.delegationHint } : {}),
    }));
  }
  if (w.routes !== undefined) {
    out.routes = w.routes.map((r) => ({
      from_agent_id: r.fromAgentId,
      to_agent_id: r.toAgentId,
      when: r.when,
    }));
  }
  if (w.canvas !== undefined) out.canvas = w.canvas;
  return out;
}

export const alloyApi = {
  // === Agent Alloy Workflows ===

  async listAlloyWorkflows(): Promise<{ workflows: AlloyWorkflow[] }> {
    const response = await apiRequest<{ workflows: ServerWorkflow[] }>(
      '/api/alloy/workflows'
    );
    return { workflows: response.workflows.map(deserializeWorkflow) };
  },

  async getAlloyWorkflow(id: string): Promise<{ workflow: AlloyWorkflow }> {
    const response = await apiRequest<{ workflow: ServerWorkflow }>(
      `/api/alloy/workflows/${encodeURIComponent(id)}`
    );
    return { workflow: deserializeWorkflow(response.workflow) };
  },

  async createAlloyWorkflow(workflow: AlloyWorkflowCreate): Promise<{ workflow: AlloyWorkflow }> {
    const response = await apiRequest<{ workflow: ServerWorkflow }>(
      '/api/alloy/workflows',
      { method: 'POST', body: JSON.stringify(serializeWorkflow(workflow)) }
    );
    return { workflow: deserializeWorkflow(response.workflow) };
  },

  async updateAlloyWorkflow(id: string, patch: AlloyWorkflowUpdate): Promise<{ workflow: AlloyWorkflow }> {
    const response = await apiRequest<{ workflow: ServerWorkflow }>(
      `/api/alloy/workflows/${encodeURIComponent(id)}`,
      { method: 'PATCH', body: JSON.stringify(serializeWorkflow(patch)) }
    );
    return { workflow: deserializeWorkflow(response.workflow) };
  },

  async deleteAlloyWorkflow(id: string): Promise<{ deleted: boolean }> {
    return apiRequest(`/api/alloy/workflows/${encodeURIComponent(id)}`, {
      method: 'DELETE',
    });
  },
};
