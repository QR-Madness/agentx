/**
 * OrgManager — the Organization surface: the whole governed space in one view
 * (ambassador apex, managers, teams, independent agents) with a dock for
 * editing a team or inspecting an agent. It is the structure surface; the
 * profile editor stays the identity surface and is one click away.
 *
 * Wide screens get the chart beside the dock. Narrow ones get the same
 * structure as an outline, and the dock takes the screen when something is
 * selected. Everything is derived from the profile and team stores
 * (`lib/orgChart`); this surface stores nothing of its own.
 *
 * Full-screen surface (ModalPortal's FULLSCREEN_SURFACES): it renders its own
 * backdrop, container and close affordances.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { ChevronLeft, ListTree, Network, Plus, Search, X } from 'lucide-react';
import { useAgentProfile } from '../../contexts/AgentProfileContext';
import { useAlloyWorkflow } from '../../contexts/AlloyWorkflowContext';
import { useModal } from '../../contexts/ModalContext';
import type { AgentProfile } from '../../lib/api';
import { buildOrgModel, layoutOrgChart, type OrgIssue } from '../../lib/orgChart';
import { orgPlacement } from '../../lib/orgPlacement';
import { SURFACES } from '../../lib/surfaces';
import { Button } from '../ui/Button';
import { useConfirm } from '../ui/ConfirmDialog';
import { Input } from '../ui/Field';
import { IconButton } from '../ui/IconButton';
import { AgentPanel } from './AgentPanel';
import { OrgCanvas } from './OrgCanvas';
import { OrgOutline } from './OrgOutline';
import { OrgOverview } from './OrgOverview';
import { TeamEditor, type TeamPayload } from './TeamEditor';
import type { OrgSelection } from './types';
import './OrgManager.css';

interface OrgManagerProps {
  onClose: () => void;
  /** Open with this team's editor showing. */
  initialTeamId?: string;
  /** Open with this agent's panel showing. */
  initialAgentId?: string;
}

// Below this the chart gives way to the outline, and the dock goes full-width.
const NARROW_QUERY = '(max-width: 860px)';
// The outline sits beside the chart by default only when there is room for all three panes.
const WIDE_QUERY = '(min-width: 1440px)';

function useMediaQuery(query: string): boolean {
  const [matches, setMatches] = useState(
    () => typeof window !== 'undefined' && window.matchMedia(query).matches,
  );
  useEffect(() => {
    const mq = window.matchMedia(query);
    const handler = (e: MediaQueryListEvent) => setMatches(e.matches);
    setMatches(mq.matches);
    mq.addEventListener('change', handler);
    return () => mq.removeEventListener('change', handler);
  }, [query]);
  return matches;
}

function plural(count: number, noun: string): string {
  return `${count} ${noun}${count === 1 ? '' : 's'}`;
}

export function OrgManager({ onClose, initialTeamId, initialAgentId }: OrgManagerProps) {
  const { profiles } = useAgentProfile();
  const { workflows, createWorkflow, updateWorkflow, deleteWorkflow } = useAlloyWorkflow();
  const { openModal, modals } = useModal();
  const confirm = useConfirm();

  const narrow = useMediaQuery(NARROW_QUERY);
  const wide = useMediaQuery(WIDE_QUERY);
  const [outlinePinned, setOutlinePinned] = useState<boolean | null>(null);
  const showOutline = narrow || (outlinePinned ?? wide);

  const [selection, setSelection] = useState<OrgSelection>(
    initialTeamId
      ? { kind: 'team', teamId: initialTeamId }
      : initialAgentId
        ? { kind: 'agent', agentId: initialAgentId }
        : { kind: 'none' },
  );
  const [query, setQuery] = useState('');
  const needle = query.trim().toLowerCase();

  const model = useMemo(() => buildOrgModel(profiles, workflows), [profiles, workflows]);
  const layout = useMemo(() => layoutOrgChart(model), [model]);
  const profilesByAgentId = useMemo(() => {
    const map = new Map<string, AgentProfile>();
    for (const p of profiles) map.set(p.agentId, p);
    return map;
  }, [profiles]);

  // Unsaved edits live in the team editor; it reports them here so every way
  // out of the editor asks first.
  const dirtyRef = useRef(false);
  const onDirtyChange = useCallback((dirty: boolean) => {
    dirtyRef.current = dirty;
  }, []);

  const editingName =
    selection.kind === 'team'
      ? (workflows.find((w) => w.id === selection.teamId)?.name ?? 'this team')
      : 'the new team';

  const mayLeaveEditor = useCallback(async (): Promise<boolean> => {
    if (!dirtyRef.current) return true;
    const ok = await confirm({
      title: 'Discard unsaved changes?',
      body: `Your edits to ${editingName} have not been saved.`,
      confirmLabel: 'Discard',
      cancelLabel: 'Keep editing',
      danger: true,
    });
    if (ok) dirtyRef.current = false;
    return ok;
  }, [confirm, editingName]);

  const select = useCallback(
    async (next: OrgSelection) => {
      if (await mayLeaveEditor()) setSelection(next);
    },
    [mayLeaveEditor],
  );
  const selectAgent = useCallback((agentId: string) => void select({ kind: 'agent', agentId }), [select]);
  const selectTeam = useCallback((teamId: string) => void select({ kind: 'team', teamId }), [select]);
  const clearSelection = useCallback(() => void select({ kind: 'none' }), [select]);
  const newTeam = useCallback(
    (managerAgentId?: string) => void select({ kind: 'new-team', managerAgentId }),
    [select],
  );

  const close = useCallback(async () => {
    if (await mayLeaveEditor()) onClose();
  }, [mayLeaveEditor, onClose]);

  // Escape closes this surface only while it is the top one, and only when
  // nothing above it (a select, a confirm) has already taken the key.
  const isTop = modals[modals.length - 1]?.component === 'orgManager';
  const isTopRef = useRef(isTop);
  isTopRef.current = isTop;
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key !== 'Escape' || e.defaultPrevented || !isTopRef.current) return;
      e.preventDefault();
      void close();
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [close]);

  useEffect(() => {
    document.body.style.overflow = 'hidden';
    return () => {
      document.body.style.overflow = '';
    };
  }, []);

  // A selection can outlive its subject (a team deleted elsewhere, a profile removed).
  const selectedTeam =
    selection.kind === 'team' ? (workflows.find((w) => w.id === selection.teamId) ?? null) : null;
  const selectedProfile =
    selection.kind === 'agent' ? (profilesByAgentId.get(selection.agentId) ?? null) : null;
  useEffect(() => {
    if (selection.kind === 'team' && !selectedTeam) setSelection({ kind: 'none' });
    if (selection.kind === 'agent' && !selectedProfile) setSelection({ kind: 'none' });
  }, [selection.kind, selectedTeam, selectedProfile]);

  const existingIds = useMemo(() => new Set(workflows.map((w) => w.id)), [workflows]);

  const saveTeam = useCallback(
    async (payload: TeamPayload, isUpdate: boolean) => {
      if (isUpdate) {
        const { id, ...patch } = payload;
        await updateWorkflow(id, patch);
        dirtyRef.current = false;
      } else {
        const created = await createWorkflow(payload);
        dirtyRef.current = false;
        setSelection({ kind: 'team', teamId: created.id });
      }
    },
    [createWorkflow, updateWorkflow],
  );

  const removeTeam = useCallback(
    async (teamId: string, teamName: string) => {
      const ok = await confirm({
        title: `Delete the ${teamName} team?`,
        body: 'Its lead and members stay as agents; only the team is removed. This cannot be undone.',
        confirmLabel: 'Delete team',
        danger: true,
      });
      if (!ok) return;
      await deleteWorkflow(teamId);
      dirtyRef.current = false;
      setSelection({ kind: 'none' });
    },
    [confirm, deleteWorkflow],
  );

  const editProfile = useCallback(
    (profileId: string) => {
      openModal({ ...SURFACES.profileEditor, props: { initialProfileId: profileId } });
    },
    [openModal],
  );
  const openDossier = useCallback(
    (profileId: string) => {
      openModal({ ...SURFACES.roster, props: { initialAgentId: profileId } });
    },
    [openModal],
  );

  const openIssue = useCallback(
    (issue: OrgIssue) => {
      const onAgent = issue.kind === 'role-not-applied' || issue.kind === 'role-without-place';
      if (onAgent && issue.agentId) selectAgent(issue.agentId);
      else if (issue.teamId) selectTeam(issue.teamId);
    },
    [selectAgent, selectTeam],
  );

  const dock =
    selection.kind === 'new-team' ? (
      <>
        <div className="org-dock-head">
          <h2 className="org-panel-title">New team</h2>
        </div>
        <TeamEditor
          key={`new:${selection.managerAgentId ?? ''}`}
          team={null}
          defaultManagerAgentId={selection.managerAgentId}
          profiles={profiles}
          model={model}
          existingIds={existingIds}
          onSave={saveTeam}
          onDirtyChange={onDirtyChange}
          onOpenAgent={selectAgent}
        />
      </>
    ) : selectedTeam ? (
      <>
        <div className="org-dock-head">
          <h2 className="org-panel-title">{selectedTeam.name}</h2>
          <span className="org-panel-id">{selectedTeam.id}</span>
        </div>
        <TeamEditor
          // Remount on save so the form starts again from what the server stored.
          key={`${selectedTeam.id}:${selectedTeam.updatedAt ?? ''}`}
          team={selectedTeam}
          profiles={profiles}
          model={model}
          existingIds={existingIds}
          onSave={saveTeam}
          onDelete={() => void removeTeam(selectedTeam.id, selectedTeam.name)}
          onDirtyChange={onDirtyChange}
          onOpenAgent={selectAgent}
        />
      </>
    ) : selectedProfile ? (
      <AgentPanel
        profile={selectedProfile}
        placement={orgPlacement(selectedProfile.agentId, profiles, workflows)}
        tier={model.tierByAgentId.get(selectedProfile.agentId)}
        issues={model.issues.filter(
          (i) =>
            i.agentId === selectedProfile.agentId &&
            (i.kind === 'role-not-applied' || i.kind === 'role-without-place'),
        )}
        onOpenTeam={selectTeam}
        onOpenAgent={selectAgent}
        onNewTeam={newTeam}
        onEditProfile={editProfile}
        onOpenDossier={openDossier}
      />
    ) : (
      <OrgOverview model={model} onOpenIssue={openIssue} onNewTeam={() => newTeam()} />
    );

  const hasSelection = selection.kind !== 'none';
  const showStructure = !narrow || !hasSelection;
  const showDock = !narrow || hasSelection;
  const agentCount = profiles.filter((p) => p.kind === 'agent').length;

  // The profile editor and the Roster open above this surface and are partly
  // see-through, so step out of sight while one is up; the state stays mounted
  // and is there again when it closes.
  const coveredStyle = isTop ? undefined : ({ visibility: 'hidden' } as const);

  return (
    <>
      <div className="org-backdrop" style={coveredStyle} onClick={() => void close()} aria-hidden />
      <div
        className="org-container"
        style={coveredStyle}
        role="dialog"
        aria-label="Organization"
        aria-modal="true"
      >
        <header className="org-header">
          <div className="org-title-group">
            <h1 className="org-title">
              <Network size={18} className="org-title-icon" />
              Organization
            </h1>
            <span className="org-title-count">
              {plural(model.managers.length, 'manager')} · {plural(model.teams.length, 'team')} ·{' '}
              {plural(agentCount, 'agent')}
            </span>
          </div>
          <div className="org-header-actions">
            <Input
              className="org-search"
              icon={<Search size={14} />}
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Find an agent or team…"
              aria-label="Find an agent or team"
            />
            {!narrow && (
              <IconButton
                active={showOutline}
                aria-label={showOutline ? 'Hide the outline' : 'Show the outline'}
                title={showOutline ? 'Hide the outline' : 'Show the outline'}
                onClick={() => setOutlinePinned(!showOutline)}
              >
                <ListTree size={16} />
              </IconButton>
            )}
            <Button variant="primary" size="sm" className="org-new" onClick={() => newTeam()}>
              <Plus size={15} />
              <span>New team</span>
            </Button>
            <IconButton aria-label="Close" title="Close" onClick={() => void close()}>
              <X size={18} />
            </IconButton>
          </div>
        </header>

        <div className={`org-body${narrow ? ' is-narrow' : ''}`}>
          {showStructure && showOutline && (
            <aside className="org-rail">
              <OrgOutline
                model={model}
                selection={selection}
                query={needle}
                onSelectAgent={selectAgent}
                onSelectTeam={selectTeam}
              />
            </aside>
          )}

          {showStructure && !narrow && (
            <OrgCanvas
              model={model}
              layout={layout}
              profilesByAgentId={profilesByAgentId}
              selection={selection}
              query={needle}
              onSelectAgent={selectAgent}
              onSelectTeam={selectTeam}
              onClear={clearSelection}
            />
          )}

          {showDock && (
            <section className="org-dock" aria-label="Details">
              {narrow && (
                <Button variant="ghost" size="sm" className="org-back" onClick={clearSelection}>
                  <ChevronLeft size={18} />
                  Organization
                </Button>
              )}
              <div className="org-dock-scroll">{dock}</div>
            </section>
          )}
        </div>
      </div>
    </>
  );
}
