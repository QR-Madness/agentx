/**
 * OrgCanvas — the organization drawn as a chart (React Flow, read-only in this
 * version: boxes select, they don't move or connect). Positions come from
 * `layoutOrgChart`, so the chart is the same every time for the same org.
 *
 * Every box is a real button, so the chart works from the keyboard; React
 * Flow's own node focus and selection are switched off in its favour.
 */

import { memo, useEffect, useMemo, useRef, useState } from 'react';
import {
  Background, Controls, Handle, Position, ReactFlow, ReactFlowProvider, useReactFlow,
  type BuiltInEdge, type Node, type NodeProps,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import { AlertTriangle, UserRound, Users } from 'lucide-react';
import type { AgentProfile } from '../../lib/api';
import { modelShortLabel } from '../../lib/modelLabel';
import { CHART, type ChartNode, type OrgChartLayout, type OrgModel } from '../../lib/orgChart';
import { AgentAvatar } from '../common/AgentAvatar';
import { TierChip, type ChipTier } from './TierChip';
import type { OrgSelection } from './types';

type AgentKind = 'ambassador' | 'manager' | 'lead' | 'member' | 'independent';

type AgentNodeData = {
  profile: AgentProfile;
  kind: AgentKind;
  selected: boolean;
  dimmed: boolean;
  flagged: boolean;
  onSelect: (agentId: string) => void;
};

type GroupNodeData = {
  kind: 'team' | 'lane';
  label: string;
  count: number;
  teamId?: string;
  selected: boolean;
  dimmed: boolean;
  flagged: boolean;
  empty: boolean;
  onSelect?: (teamId: string) => void;
};

type BandNodeData = { label: string };

// Where the lead → member "spine" leaves the lead: with the edge's step offset
// below, it runs straight down and turns right into each member.
const SPINE_X = 11;
const SPINE_OFFSET = CHART.memberIndent - SPINE_X;

const CHIP: Partial<Record<AgentKind, ChipTier>> = {
  ambassador: 'ambassador',
  manager: 'manager',
  lead: 'lead',
};

const AgentNode = memo(function AgentNode({ data }: NodeProps<Node<AgentNodeData>>) {
  const { profile, kind, selected, dimmed, flagged, onSelect } = data;
  const chip = CHIP[kind];
  const cls = [
    'org-node',
    `org-node--${kind}`,
    selected ? 'is-selected' : '',
    dimmed ? 'is-dimmed' : '',
  ].join(' ');
  return (
    <>
      <Handle id="t" type="target" position={Position.Top} className="org-handle" isConnectable={false} />
      <Handle id="l" type="target" position={Position.Left} className="org-handle" isConnectable={false} />
      <button
        type="button"
        className={cls}
        aria-pressed={selected}
        aria-label={`${profile.name}, ${kind}${flagged ? ', needs attention' : ''}`}
        onClick={() => onSelect(profile.agentId)}
      >
        <span className="org-node-avatar" aria-hidden>
          <AgentAvatar avatar={profile.avatar} size={18} fill />
        </span>
        <span className="org-node-text">
          <span className="org-node-name">
            <span className="org-node-name-text">{profile.name}</span>
            {chip && <TierChip tier={chip} />}
          </span>
          <span className="org-node-sub">{modelShortLabel(profile.defaultModel) ?? 'default model'}</span>
        </span>
        {flagged && (
          <span className="org-node-flag" title="Needs attention">
            <AlertTriangle size={13} />
          </span>
        )}
      </button>
      <Handle id="b" type="source" position={Position.Bottom} className="org-handle" isConnectable={false} />
      <Handle
        id="s"
        type="source"
        position={Position.Bottom}
        className="org-handle"
        style={{ left: SPINE_X }}
        isConnectable={false}
      />
    </>
  );
});

const GroupNode = memo(function GroupNode({ data }: NodeProps<Node<GroupNodeData>>) {
  const { kind, label, count, teamId, selected, dimmed, flagged, empty, onSelect } = data;
  const cls = [
    'org-group',
    `org-group--${kind}`,
    selected ? 'is-selected' : '',
    dimmed ? 'is-dimmed' : '',
  ].join(' ');
  const head = (
    <>
      {kind === 'team' ? <Users size={13} /> : <UserRound size={13} />}
      <span className="org-group-name">{label}</span>
      <span className="org-group-count">{count}</span>
      {flagged && <AlertTriangle size={13} className="org-group-flag" />}
    </>
  );
  return (
    <div className={cls}>
      {kind === 'team' && teamId && onSelect ? (
        <button
          type="button"
          className="org-group-head"
          aria-pressed={selected}
          aria-label={`${label} team, ${count} ${count === 1 ? 'agent' : 'agents'}`}
          onClick={() => onSelect(teamId)}
        >
          {head}
        </button>
      ) : (
        <div className="org-group-head org-group-head--static">{head}</div>
      )}
      {empty && <span className="org-group-empty">No lead or member resolves.</span>}
    </div>
  );
});

const BandNode = memo(function BandNode({ data }: NodeProps<Node<BandNodeData>>) {
  return <div className="org-band">{data.label}</div>;
});

const nodeTypes = { orgAgent: AgentNode, orgGroup: GroupNode, orgBand: BandNode };

interface OrgCanvasProps {
  model: OrgModel;
  layout: OrgChartLayout;
  profilesByAgentId: Map<string, AgentProfile>;
  selection: OrgSelection;
  /** Lower-cased search text; boxes that don't match are dimmed. */
  query: string;
  onSelectAgent: (agentId: string) => void;
  onSelectTeam: (teamId: string) => void;
  onClear: () => void;
}

function agentMatches(profile: AgentProfile, query: string): boolean {
  return (
    profile.name.toLowerCase().includes(query) ||
    profile.agentId.toLowerCase().includes(query) ||
    (profile.delegationHint ?? '').toLowerCase().includes(query)
  );
}

function CanvasInner({
  model, layout, profilesByAgentId, selection, query, box, onSelectAgent, onSelectTeam, onClear,
}: OrgCanvasProps & { box: string }) {
  const { fitView } = useReactFlow();

  // A role problem is fixed on the agent; a broken reference or an empty team on the team.
  const flaggedAgents = useMemo(
    () =>
      new Set(
        model.issues
          .filter((i) => i.kind === 'role-not-applied' || i.kind === 'role-without-place')
          .map((i) => i.agentId as string),
      ),
    [model.issues],
  );
  const flaggedTeams = useMemo(
    () =>
      new Set(
        model.issues
          .filter((i) => i.kind === 'unresolved-agent' || i.kind === 'team-empty')
          .map((i) => i.teamId as string),
      ),
    [model.issues],
  );
  const teamsById = useMemo(() => new Map(model.teams.map((t) => [t.id, t])), [model.teams]);

  const nodes = useMemo<Node[]>(() => {
    const selectedAgent = selection.kind === 'agent' ? selection.agentId : null;
    const selectedTeam = selection.kind === 'team' ? selection.teamId : null;
    const out: Node[] = [];

    const base = (n: ChartNode) => ({
      id: n.id,
      position: { x: n.x, y: n.y },
      width: n.width,
      height: n.height,
      style: { width: n.width, height: n.height },
      draggable: false,
      selectable: false,
      ...(n.parentId ? { parentId: n.parentId, extent: 'parent' as const } : {}),
    });

    for (const n of layout.nodes) {
      if (n.kind === 'band') {
        out.push({ ...base(n), type: 'orgBand', data: { label: n.label ?? '' } satisfies BandNodeData });
      } else if (n.kind === 'team' || n.kind === 'lane') {
        const team = n.teamId ? teamsById.get(n.teamId) : undefined;
        const count = team
          ? team.members.length + (team.lead ? 1 : 0)
          : model.independents.length;
        const nameHit = Boolean(query) && (n.label ?? '').toLowerCase().includes(query);
        const childHit =
          Boolean(query) &&
          (team
            ? [team.lead, ...team.members].some((p) => p && agentMatches(p, query))
            : model.independents.some((p) => agentMatches(p, query)));
        out.push({
          ...base(n),
          type: 'orgGroup',
          data: {
            kind: n.kind,
            label: n.label ?? '',
            count,
            teamId: n.teamId,
            selected: n.teamId !== undefined && n.teamId === selectedTeam,
            dimmed: Boolean(query) && !nameHit && !childHit,
            flagged: n.teamId !== undefined && flaggedTeams.has(n.teamId),
            empty: count === 0,
            onSelect: onSelectTeam,
          } satisfies GroupNodeData,
        });
      } else if (n.agentId) {
        const profile = profilesByAgentId.get(n.agentId);
        if (!profile) continue;
        const team = n.teamId ? teamsById.get(n.teamId) : undefined;
        const teamHit = Boolean(query) && Boolean(team) && team!.name.toLowerCase().includes(query);
        out.push({
          ...base(n),
          type: 'orgAgent',
          data: {
            profile,
            kind: n.kind,
            selected: n.agentId === selectedAgent,
            dimmed: Boolean(query) && !teamHit && !agentMatches(profile, query),
            flagged: flaggedAgents.has(n.agentId),
            onSelect: onSelectAgent,
          } satisfies AgentNodeData,
        });
      }
    }
    return out;
  }, [layout.nodes, model.independents, teamsById, profilesByAgentId, selection, query, flaggedAgents, flaggedTeams, onSelectAgent, onSelectTeam]);

  const edges = useMemo<BuiltInEdge[]>(() => {
    const selectedAgent = selection.kind === 'agent' ? selection.agentId : null;
    const agentOf = new Map(layout.nodes.map((n) => [n.id, n.agentId]));
    return layout.edges.map((e) => {
      const active =
        selectedAgent !== null &&
        (agentOf.get(e.source) === selectedAgent || agentOf.get(e.target) === selectedAgent);
      return {
        id: e.id,
        source: e.source,
        target: e.target,
        sourceHandle: e.kind === 'works' ? 's' : 'b',
        targetHandle: e.kind === 'works' ? 'l' : 't',
        type: 'smoothstep',
        pathOptions: e.kind === 'works' ? { offset: SPINE_OFFSET, borderRadius: 6 } : { borderRadius: 10 },
        className: `org-edge org-edge--${e.kind}${active ? ' is-active' : ''}`,
        focusable: false,
        selectable: false,
      } satisfies BuiltInEdge;
    });
  }, [layout.edges, layout.nodes, selection]);

  // Keep the whole chart in view when it changes size or the space around it
  // does (the window, or the outline beside it).
  useEffect(() => {
    void fitView({ padding: 0.1, duration: 180 });
  }, [fitView, box, layout.width, layout.height]);

  return (
    <ReactFlow
      nodes={nodes}
      edges={edges}
      nodeTypes={nodeTypes}
      fitView
      fitViewOptions={{ padding: 0.1 }}
      minZoom={0.3}
      maxZoom={1.6}
      nodesDraggable={false}
      nodesConnectable={false}
      nodesFocusable={false}
      edgesFocusable={false}
      elementsSelectable={false}
      onPaneClick={onClear}
      proOptions={{ hideAttribution: true }}
    >
      <Background gap={24} />
      <Controls showInteractive={false} />
    </ReactFlow>
  );
}

export function OrgCanvas(props: OrgCanvasProps) {
  const wrapperRef = useRef<HTMLDivElement>(null);
  const [box, setBox] = useState('');
  useEffect(() => {
    const el = wrapperRef.current;
    if (!el || typeof ResizeObserver === 'undefined') return;
    const observer = new ResizeObserver(([entry]) => {
      setBox(`${Math.round(entry.contentRect.width)}x${Math.round(entry.contentRect.height)}`);
    });
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  return (
    <div ref={wrapperRef} className="org-canvas" role="region" aria-label="Organization chart">
      <ReactFlowProvider>
        <CanvasInner {...props} box={box} />
      </ReactFlowProvider>
    </div>
  );
}
