import { describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { TeamEditor, type TeamPayload } from './TeamEditor';
import { buildOrgModel } from '../../lib/orgChart';
import type { AgentProfile, AlloyWorkflow, AlloyWorkflowMember } from '../../lib/api';

// One manager with two teams, one team outside the organization, one agent on
// no team, one ambassador — the same shape as lib/orgChart.test.ts.
function agent(agentId: string, kind: 'agent' | 'ambassador' = 'agent'): AgentProfile {
  return { id: agentId, name: agentId.toUpperCase(), kind, agentId, orgLevel: 'agent' } as AgentProfile;
}

function team(
  id: string,
  supervisorAgentId: string,
  members: Array<[string, string?]>,
  managerAgentId?: string,
): AlloyWorkflow {
  const list: AlloyWorkflowMember[] = [
    { agentId: supervisorAgentId, role: 'supervisor' },
    ...members.map(([agentId, delegationHint]) => ({ agentId, role: 'specialist' as const, delegationHint })),
  ];
  return { id, name: id.toUpperCase(), supervisorAgentId, managerAgentId, members: list } as AlloyWorkflow;
}

const profiles = ['m', 'l1', 'l2', 'l3', 'a1', 'a2', 'a3', 'a4', 'free'].map((id) => agent(id));
profiles.push(agent('amb', 'ambassador'));

const workflows = [
  team('t1', 'l1', [['a1', 'hint A'], ['a2']], 'm'),
  team('t2', 'l2', [['a3']], 'm'),
  team('t3', 'l3', [['a4']]),
];
const model = buildOrgModel(profiles, workflows);

function renderEditor(over: Partial<Parameters<typeof TeamEditor>[0]> = {}) {
  const props = {
    team: workflows[0],
    profiles,
    model,
    existingIds: new Set(workflows.map((w) => w.id)),
    onSave: vi.fn<(p: TeamPayload, isUpdate: boolean) => Promise<void>>().mockResolvedValue(undefined),
    onDirtyChange: vi.fn(),
    onOpenAgent: vi.fn(),
    ...over,
  };
  render(<TeamEditor {...props} />);
  return props;
}

describe('TeamEditor', () => {
  it("shows an existing team's lead and manager by name, and no ID field", () => {
    renderEditor();
    expect(screen.getByLabelText('Lead agent')).toHaveTextContent('L1');
    expect(screen.getByLabelText('Owning manager')).toHaveTextContent('M');
    expect(screen.queryByLabelText('ID')).toBeNull();
    expect(screen.getByDisplayValue('T1')).toBeInTheDocument();
  });

  it('keeps Save disabled until something changes, then reports unsaved edits', async () => {
    const props = renderEditor();
    const save = screen.getByRole('button', { name: 'Save changes' });
    expect(save).toBeDisabled();
    expect(props.onDirtyChange).toHaveBeenLastCalledWith(false);

    await userEvent.type(screen.getByLabelText('When to delegate to A2'), 'x');
    expect(save).toBeEnabled();
    expect(props.onDirtyChange).toHaveBeenLastCalledWith(true);
  });

  it('saves the lead first, then the members with trimmed hints, and null for no manager', async () => {
    const props = renderEditor({ team: workflows[2] });
    await userEvent.type(screen.getByLabelText('When to delegate to A4'), '  long reports  ');
    await userEvent.click(screen.getByRole('button', { name: 'Save changes' }));

    expect(props.onSave).toHaveBeenCalledWith(
      {
        id: 't3',
        name: 'T3',
        description: undefined,
        supervisorAgentId: 'l3',
        managerAgentId: null,
        members: [
          { agentId: 'l3', role: 'supervisor' },
          { agentId: 'a4', role: 'specialist', delegationHint: 'long reports' },
        ],
      },
      true,
    );
  });

  it('checks a new team before saving: a name, a free ID, and a lead', async () => {
    const props = renderEditor({ team: null });
    const create = screen.getByRole('button', { name: 'Create team' });

    await userEvent.click(create);
    expect(screen.getByRole('alert')).toHaveTextContent('Name is required.');

    await userEvent.type(screen.getByLabelText('Name'), 'T1');
    expect(screen.getByLabelText('ID')).toHaveValue('t1');
    await userEvent.click(create);
    expect(screen.getByRole('alert')).toHaveTextContent('A team with id "t1" already exists.');

    await userEvent.type(screen.getByLabelText('Name'), ' Squad');
    expect(screen.getByLabelText('ID')).toHaveValue('t1-squad');
    await userEvent.click(create);
    expect(screen.getByRole('alert')).toHaveTextContent('Pick a lead agent.');
    expect(props.onSave).not.toHaveBeenCalled();
  });

  it('does not count a derived ID as an edit of a blank new team', () => {
    const props = renderEditor({ team: null });
    expect(props.onDirtyChange).toHaveBeenLastCalledWith(false);
  });

  it('pre-selects the manager a new team was started from', () => {
    renderEditor({ team: null, defaultManagerAgentId: 'm' });
    expect(screen.getByLabelText('Owning manager')).toHaveTextContent('M');
  });

  it('offers only agents not already placed on this team, and says where each one sits', async () => {
    renderEditor();
    await userEvent.click(screen.getByRole('button', { name: 'Add member' }));
    const picker = screen.getByRole('group', { name: 'Add member' });
    const names = within(picker)
      .getAllByRole('button')
      .map((b) => b.textContent)
      .filter((t) => t);

    expect(names).toEqual(['L2Leads T2', 'L3Leads T3', 'A3On T2', 'A4On T3', 'FREE']);
  });

  it('adds members from the picker until it is closed with Escape', async () => {
    renderEditor();
    await userEvent.click(screen.getByRole('button', { name: 'Add member' }));
    const picker = screen.getByRole('group', { name: 'Add member' });

    await userEvent.click(within(picker).getByRole('button', { name: /^A3/ }));
    expect(screen.getByRole('button', { name: 'Remove A3' })).toBeInTheDocument();
    expect(within(picker).queryByRole('button', { name: /^A3/ })).toBeNull();

    const search = within(picker).getByLabelText('Search agents to add');
    const outer = vi.fn();
    window.addEventListener('keydown', outer);
    fireEvent.keyDown(search, { key: 'Escape' });
    window.removeEventListener('keydown', outer);
    expect(screen.queryByRole('group', { name: 'Add member' })).toBeNull();
    expect(outer).not.toHaveBeenCalled();
  });

  it('removes a member and opens a member by its name', async () => {
    const props = renderEditor();
    await userEvent.click(screen.getByRole('button', { name: 'A1' }));
    expect(props.onOpenAgent).toHaveBeenCalledWith('a1');

    await userEvent.click(screen.getByRole('button', { name: 'Remove A1' }));
    expect(screen.queryByRole('button', { name: 'A1' })).toBeNull();
  });

  it('offers Delete only for an existing team', async () => {
    const onDelete = vi.fn();
    renderEditor({ onDelete });
    await userEvent.click(screen.getByRole('button', { name: 'Delete team' }));
    expect(onDelete).toHaveBeenCalled();
  });

  it('says when a member id no longer resolves to an agent', () => {
    renderEditor({ team: team('t9', 'l1', [['ghost']]) });
    expect(screen.getByText(/No agent has this id any more/)).toBeInTheDocument();
  });
});
