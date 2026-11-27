/**
 * OrgCanvas — the organization drawn as a chart (React Flow). Positions come from
 * `layoutOrgChart`, so the chart is the same every time for the same org, except
 * where the user dragged a top-level box (an ambassador, a manager, a whole team):
 * those come from the stored layout (`layoutKeyOf`). Nothing connects.
 *
 * Every box is a real button, so the chart works from the keyboard; React
 * Flow's own node focus and selection are switched off in its favour.
 */

import { memo, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  Background, Controls, Handle, MarkerType, Position, ReactFlow, ReactFlowProvider, useReactFlow,
  type BuiltInEdge, type Node, type NodeChangeset, type NodeProps, type XYPosition,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import { AlertTriangle, UserRound, Users } from 'lucide-react';
import type { AgentProfile, LayoutPositions } from '../../lib/api';
import { modelShortLabel } from '../../lib/modelLabel';
import {
  CHART, layoutKeyOf, type ChartNode, type OrgChartLayout, type OrgModel,
} from '../../lib/orgChart';
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

// Room around the chart when it is fitted to the view, on each side. React Flow 13
// reads a bare number as pixels; this is its own default since then, and what the
// fractional `0.1` gave before (about 4.5% a side).
const FIT_PADDING = '5%';

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
  /** A line was clicked: explain it in the dock (read-only). */
  onSelectEdge?: (edgeId: string) => void;
  /** Stored positions of dragged boxes, by `layoutKeyOf` key. */
  positions: LayoutPositions;
  /** Called when a drag ends; without it nothing moves (no layout store to save to). */
  onMoveNode?: (key: string, position: XYPosition) => void;
  /** Changes when the stored layout loads or resets: the chart refits (never after a drag). */
  fitKey: number;
}

function agentMatches(profile: AgentProfile, query: string): boolean {
  return (
    profile.name.toLowerCase().includes(query) ||
    profile.agentId.toLowerCase().includes(query) ||
    (profile.delegationHint ?? '').toLowerCase().includes(query)
  );
}

function CanvasInner({
  model, layout, profilesByAgentId, selection, query, box, onSelectAgent, onSelectTeam, onClear, onSelectEdge,
  positions, onMoveNode, fitKey,
}: OrgCanvasProps & { box: string }) {
  const { fitView } = useReactFlow();

  // Where a box is while it's being dragged (by node id); committed on drop.
  const [dragging, setDragging] = useState<Record<string, XYPosition>>({});
  // The click that ends a drag lands on the box's button; it mustn't also select it.
  const justDragged = useRef(false);
  const selectAgent = useCallback(
    (agentId: string) => { if (!justDragged.current) onSelectAgent(agentId); },
    [onSelectAgent],
  );
  const selectTeam = useCallback(
    (teamId: string) => { if (!justDragged.current) onSelectTeam(teamId); },
    [onSelectTeam],
  );

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

    const base = (n: ChartNode) => {
      const key = onMoveNode ? layoutKeyOf(n) : null;
      return {
        id: n.id,
        position: dragging[n.id] ?? (key ? positions[key] : undefined) ?? { x: n.x, y: n.y },
        width: n.width,
        height: n.height,
        style: { width: n.width, height: n.height },
        draggable: key !== null,
        selectable: false,
        ...(n.parentId ? { parentId: n.parentId, extent: 'parent' as const } : {}),
      };
    };

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
            onSelect: selectTeam,
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
            onSelect: selectAgent,
          } satisfies AgentNodeData,
        });
      }
    }
    return out;
  }, [layout.nodes, model.independents, teamsById, profilesByAgentId, selection, query, flaggedAgents, flaggedTeams, selectAgent, selectTeam, onMoveNode, positions, dragging]);

  const edges = useMemo<BuiltInEdge[]>(() => {
    const selectedAgent = selection.kind === 'agent' ? selection.agentId : null;
    const agentOf = new Map(layout.nodes.map((n) => [n.id, n.agentId]));
    const selectedEdge = selection.kind === 'edge' ? selection.edgeId : null;
    return layout.edges.map((e) => {
      const active =
        e.id === selectedEdge ||
        (selectedAgent !== null &&
          (agentOf.get(e.source) === selectedAgent || agentOf.get(e.target) === selectedAgent));
      return {
        id: e.id,
        source: e.source,
        target: e.target,
        sourceHandle: e.kind === 'works' ? 's' : 'b',
        targetHandle: e.kind === 'works' ? 'l' : 't',
        type: 'smoothstep',
        pathOptions: e.kind === 'works' ? { offset: SPINE_OFFSET, borderRadius: 6 } : { borderRadius: 10 },
        // A member can escalate back to its lead: a chevron at the lead's end of
        // the line (a start marker points back toward the source).
        ...(e.kind === 'works'
          ? {
              markerStart: {
                type: MarkerType.Arrow,
                width: 14,
                height: 14,
                color: active ? 'var(--accent-primary)' : 'var(--border-emphasis)',
              },
            }
          : {}),
        className: `org-edge org-edge--${e.kind}${active ? ' is-active' : ''}`,
        focusable: false,
        selectable: false,
      } satisfies BuiltInEdge;
    });
  }, [layout.edges, layout.nodes, selection]);

  // Keep the whole chart in view when it changes size or the space around it
  // does (the window, or the outline beside it).
  // ...and when the stored layout loads or resets (`fitKey`), but never after a drag.
  useEffect(() => {
    void fitView({ padding: FIT_PADDING, duration: 180 });
  }, [fitView, box, layout.width, layout.height, fitKey]);

  const keyByNodeId = useMemo(() => {
    const map = new Map<string, string>();
    for (const n of layout.nodes) {
      const key = layoutKeyOf(n);
      if (key) map.set(n.id, key);
    }
    return map;
  }, [layout.nodes]);

  const onNodesChange = useCallback((changes: NodeChangeset) => {
    setDragging((prev) => {
      let next = prev;
      for (const change of changes) {
        if (change.type !== 'position') continue;
        if (next === prev) next = { ...prev };
        // Only an in-progress drag is held here; the final change (dragging: false)
        // clears it whichever order it arrives in relative to onNodeDragStop.
        if (change.dragging && change.position) next[change.id] = change.position;
        else delete next[change.id];
      }
      return next;
    });
  }, []);

  const onNodeDragStop = useCallback(
    (_event: unknown, node: Node) => {
      const key = keyByNodeId.get(node.id);
      if (key && onMoveNode) {
        onMoveNode(key, { x: Math.round(node.position.x), y: Math.round(node.position.y) });
      }
      setDragging((prev) => {
        const { [node.id]: _dropped, ...rest } = prev;
        return rest;
      });
      justDragged.current = true;
      setTimeout(() => { justDragged.current = false; }, 0);
    },
    [keyByNodeId, onMoveNode],
  );

  return (
    <ReactFlow
      nodes={nodes}
      edges={edges}
      nodeTypes={nodeTypes}
      fitView
      fitViewOptions={{ padding: FIT_PADDING }}
      // The chart's colours come from our theme tokens (`.org-canvas` in
      // OrgManager.css). Pinned to light, React Flow's own defaults stay the
      // ones every theme was designed against; left alone, v13 follows the OS.
      forceColorMode="light"
      minZoom={0.3}
      maxZoom={1.6}
      nodesDraggable={Boolean(onMoveNode)}
      onNodesChange={onNodesChange}
      onNodeDragStop={onNodeDragStop}
      nodesConnectable={false}
      nodesFocusable={false}
      edgesFocusable={false}
      elementsSelectable={false}
      onPaneClick={onClear}
      onEdgeClick={onSelectEdge ? (_event, edge) => onSelectEdge(edge.id) : undefined}
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
