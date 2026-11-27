// ─── Graph view ────────────────────────────────────────────────────────────
// The memory explore tool: a graph centered on either an entity (its related
// entities) or a procedure (its provenance — scope, agent, evidence
// conversations). As the flat list sections are retired, this becomes the
// single explorer surface.

import { useState, useEffect, useMemo, useRef } from 'react';
import { RefreshCw, GitBranch, Search } from 'lucide-react';
import {
  ReactFlow, Background, Controls, Handle, Position, BaseEdge,
  getStraightPath, useInternalNode, useNodesState, useEdgesState,
  type NodeProps, type EdgeProps, type InternalNode, type NodeOrigin, type Node, type Edge
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import '../../styles/flow.css';
import { useMemoryEntities, useEntityGraph, useMemoryProcedures } from '../../lib/hooks';
import type { MemoryProcedure } from '../../lib/api';
import { procedureHeadline } from './procedureLabel';
import { edgeEnds, ringPositions, type Box } from './graphGeometry';
import { Input } from '../ui';

// Custom node (module-level to avoid re-creation on each render). `variant`
// drives styling: entity / procedure (center) / scope / agent / conversation.
type GraphNodeData = {
  label: string;
  typeLabel: string;
  isCenter: boolean;
  variant: 'entity' | 'procedure' | 'scope' | 'agent' | 'conversation';
} & Record<string, unknown>;

function GraphNodeComponent({ data }: NodeProps) {
  const d = data as GraphNodeData;
  // React Flow draws an edge only between handles, so each node keeps a pair.
  // SpokeEdge ignores where they sit, and they are hidden (`.graph-handle`)
  // and cannot start a connection.
  return (
    <div className={`graph-node ${d.variant}${d.isCenter ? ' center' : ''}`}>
      <Handle type="target" position={Position.Left} className="graph-handle" isConnectable={false} />
      <span className="graph-node-name">{d.label}</span>
      <span className="graph-node-type">{d.typeLabel}</span>
      <Handle type="source" position={Position.Right} className="graph-handle" isConnectable={false} />
    </div>
  );
}

const nodeTypes = { graphNode: GraphNodeComponent };

/** A node's box in flow coordinates, once React Flow has measured it. */
function measuredBox(node: InternalNode | undefined): Box | null {
  const width = node?.measured.width;
  const height = node?.measured.height;
  if (!node || !width || !height) return null;
  return { ...node.internals.positionAbsolute, width, height };
}

// A straight edge between the borders its two nodes turn to each other, at any
// angle (graphGeometry.edgeEnds), with the label halfway along. It reads the
// live node boxes, so it still takes the short way after a node is dragged.
function SpokeEdge({
  id, source, target, style, interactionWidth, markerStart, markerEnd,
  label, labelStyle, labelShowBg, labelBgStyle, labelBgPadding, labelBgBorderRadius,
}: EdgeProps) {
  const sourceNode = useInternalNode(source);
  const targetNode = useInternalNode(target);
  const sourceBox = measuredBox(sourceNode);
  const targetBox = measuredBox(targetNode);
  if (!sourceBox || !targetBox) return null;
  const ends = edgeEnds(sourceBox, targetBox);
  const [path, labelX, labelY] = getStraightPath({
    sourceX: ends.source.x,
    sourceY: ends.source.y,
    targetX: ends.target.x,
    targetY: ends.target.y,
  });
  return (
    <BaseEdge
      id={id}
      path={path}
      style={style}
      interactionWidth={interactionWidth}
      markerStart={markerStart}
      markerEnd={markerEnd}
      label={label}
      labelX={labelX}
      labelY={labelY}
      labelStyle={labelStyle}
      labelShowBg={labelShowBg}
      labelBgStyle={labelBgStyle}
      labelBgPadding={labelBgPadding}
      labelBgBorderRadius={labelBgBorderRadius}
    />
  );
}

const edgeTypes = { spoke: SpokeEdge };

// Node positions are centres, so the ring sits around the middle of the centre
// node whatever the nodes' widths.
const NODE_ORIGIN: NodeOrigin = [0.5, 0.5];

// Max evidence-conversation nodes to render around a procedure (keep it legible).
const MAX_EVIDENCE_NODES = 6;

// Lay the neighbours out on a ring around the centre node, one spoke edge each.
function radialLayout(
  centerId: string,
  neighbors: Array<{ id: string; data: GraphNodeData; edgeLabel: string }>,
): { nodes: Node[]; edges: Edge[] } {
  const positions = ringPositions(neighbors.length);
  const nodes: Node[] = neighbors.map((nb, i) => ({
    id: nb.id,
    type: 'graphNode',
    data: nb.data,
    position: positions[i],
  }));
  // No `animated` edges: the constant SVG dash repaint is the usual ReactFlow
  // jank source. Static edges keep panning smooth even on low-end webviews.
  const edges: Edge[] = neighbors.map((nb, i) => ({
    id: `e-${i}`,
    type: 'spoke',
    source: centerId,
    target: nb.id,
    label: nb.edgeLabel,
  }));
  return { nodes, edges };
}

export function MemoryGraphView({ channel = '_all', refreshToken = 0 }: {
  channel?: string;
  /** Change it to refetch in place (the workbench bumps it after Consolidate, Import and Refresh). */
  refreshToken?: number;
}) {
  // Selection is mutually exclusive: an entity OR a procedure is centered.
  const [graphEntityId, setGraphEntityId] = useState<string | null>(null);
  const [graphProcedureId, setGraphProcedureId] = useState<string | null>(null);
  const [entitySearch, setEntitySearch] = useState('');

  const { entities, loading: entitiesLoading, refresh: refreshEntities } = useMemoryEntities(channel, 1, entitySearch);
  const { procedures, loading: proceduresLoading, refresh: refreshProcedures } = useMemoryProcedures(channel, 1);
  const { graph, loading: graphLoading, refresh: refreshGraph } = useEntityGraph(graphEntityId);

  // Refetch when the token changes, keeping the selection and search. The
  // token seen at mount is skipped: the hooks fetch on their own then.
  const seenRefreshToken = useRef(refreshToken);
  useEffect(() => {
    if (seenRefreshToken.current === refreshToken) return;
    seenRefreshToken.current = refreshToken;
    refreshEntities();
    refreshProcedures();
    refreshGraph();
  }, [refreshToken, refreshEntities, refreshProcedures, refreshGraph]);

  const [nodes, setNodes, onNodesChange] = useNodesState<Node>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);

  const selectedProcedure = useMemo<MemoryProcedure | null>(
    () => procedures.find(p => p.id === graphProcedureId) ?? null,
    [procedures, graphProcedureId],
  );

  // Entity-centered graph.
  useEffect(() => {
    if (graphProcedureId) return; // procedure selection owns the canvas
    if (!graph) {
      setNodes([]);
      setEdges([]);
      return;
    }
    const center: Node = {
      id: graph.entity.id,
      type: 'graphNode',
      data: { label: graph.entity.name, typeLabel: graph.entity.type, isCenter: true, variant: 'entity' },
      position: { x: 0, y: 0 },
    };
    const { nodes: neighbors, edges: relEdges } = radialLayout(
      graph.entity.id,
      graph.relationships.map(rel => ({
        id: rel.target.id,
        edgeLabel: rel.type,
        data: { label: rel.target.name, typeLabel: rel.target.type, isCenter: false, variant: 'entity' },
      })),
    );
    setNodes([center, ...neighbors]);
    setEdges(relEdges);
  }, [graph, graphProcedureId, setNodes, setEdges]);

  // Procedure-centered graph (built client-side from the procedure's fields —
  // scope, agent, and the conversations it was distilled from).
  useEffect(() => {
    if (!selectedProcedure) return;
    const p = selectedProcedure;
    const center: Node = {
      id: p.id,
      type: 'graphNode',
      data: {
        label: procedureHeadline(p.trigger, p.body),
        typeLabel: `Procedure · strength ${p.strength}`,
        isCenter: true,
        variant: 'procedure',
      },
      position: { x: 0, y: 0 },
    };

    const neighbors: Array<{ id: string; data: GraphNodeData; edgeLabel: string }> = [];
    neighbors.push({
      id: `scope:${p.id}`,
      edgeLabel: 'scope',
      data: { label: p.scope, typeLabel: 'Scope', isCenter: false, variant: 'scope' },
    });
    if (p.agent_id) {
      neighbors.push({
        id: `agent:${p.id}`,
        edgeLabel: 'corrects',
        data: { label: p.agent_id, typeLabel: 'Agent', isCenter: false, variant: 'agent' },
      });
    }
    const convRefs = (p.evidence_refs ?? [])
      .filter(r => r.startsWith('conv:'))
      .map(r => r.slice('conv:'.length))
      .slice(0, MAX_EVIDENCE_NODES);
    convRefs.forEach((cid, i) => {
      neighbors.push({
        id: `conv:${p.id}:${i}`,
        edgeLabel: 'distilled from',
        data: { label: `conv ${cid.slice(0, 8)}`, typeLabel: 'Conversation', isCenter: false, variant: 'conversation' },
      });
    });

    const { nodes: nbNodes, edges: nbEdges } = radialLayout(p.id, neighbors);
    setNodes([center, ...nbNodes]);
    setEdges(nbEdges);
  }, [selectedProcedure, setNodes, setEdges]);

  const selectEntity = (id: string) => {
    setGraphProcedureId(null);
    setGraphEntityId(graphEntityId === id ? null : id);
  };
  const selectProcedure = (id: string) => {
    setGraphEntityId(null);
    setGraphProcedureId(graphProcedureId === id ? null : id);
  };

  const hasSelection = !!graphEntityId || !!graphProcedureId;
  const canvasLoading = (!!graphEntityId && graphLoading);

  return (
    <div className="graph-view">
      <div className="graph-entity-list">
        <div className="graph-search">
          <Input
            icon={<Search size={14} />}
            placeholder="Search a topic or entity…"
            value={entitySearch}
            onChange={e => setEntitySearch(e.target.value)}
          />
        </div>
        <div className="graph-list-header">Entities</div>
        {entitiesLoading ? (
          <div className="graph-list-loading"><RefreshCw size={16} className="spin" /></div>
        ) : entities.map(e => (
          <button
            key={e.id}
            className={`graph-entity-btn${graphEntityId === e.id ? ' active' : ''}`}
            onClick={() => selectEntity(e.id)}
          >
            <span className="graph-btn-name">{e.name}</span>
            <span className="graph-btn-type">{e.type}</span>
          </button>
        ))}

        <div className="graph-list-header">Procedures</div>
        {proceduresLoading ? (
          <div className="graph-list-loading"><RefreshCw size={16} className="spin" /></div>
        ) : procedures.length === 0 ? (
          <div className="graph-list-empty">None yet</div>
        ) : procedures.map(p => (
          <button
            key={p.id}
            className={`graph-entity-btn${graphProcedureId === p.id ? ' active' : ''}`}
            onClick={() => selectProcedure(p.id)}
          >
            <span className="graph-btn-name">{procedureHeadline(p.trigger, p.body)}</span>
            <span className="graph-btn-type procedure">Procedure · {p.strength}</span>
          </button>
        ))}
      </div>
      <div className="graph-canvas">
        {!hasSelection ? (
          <div className="graph-empty">
            <GitBranch size={32} />
            <p>Select an entity or procedure to explore</p>
          </div>
        ) : canvasLoading ? (
          <div className="graph-empty"><RefreshCw size={24} className="spin" /></div>
        ) : (
          <ReactFlow
            // Colours come from the theme tokens (`.ax-flow` in styles/flow.css),
            // which also hold the canvas to the app's colour scheme rather than
            // the OS's, so it needs no `forceColorMode`.
            className="ax-flow"
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            nodeTypes={nodeTypes}
            edgeTypes={edgeTypes}
            nodeOrigin={NODE_ORIGIN}
            fitView
            onlyRenderVisibleElements
            // fitView stops at minZoom. The ring grows past ten neighbours, and
            // at the 50 the graph endpoint returns, a desktop canvas needs about
            // 0.23 to show all of it (0.3 left the rim off-screen).
            minZoom={0.2}
            maxZoom={1.6}
            proOptions={{ hideAttribution: true }}
          >
            <Background />
            <Controls />
          </ReactFlow>
        )}
      </div>
    </div>
  );
}
