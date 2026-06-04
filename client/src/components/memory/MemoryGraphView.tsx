// ─── Graph view ────────────────────────────────────────────────────────────
// The memory explore tool: a graph centered on either an entity (its related
// entities) or a procedure (its provenance — scope, agent, evidence
// conversations). As the flat list sections are retired, this becomes the
// single explorer surface.

import { useState, useEffect, useMemo } from 'react';
import { RefreshCw, GitBranch } from 'lucide-react';
import {
  ReactFlow, Background, Controls, Handle, Position,
  useNodesState, useEdgesState,
  type NodeProps, type Node, type Edge
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import { useMemoryEntities, useEntityGraph, useMemoryProcedures } from '../../lib/hooks';
import type { MemoryProcedure } from '../../lib/api';
import { procedureHeadline } from './procedureLabel';

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
  return (
    <div className={`graph-node ${d.variant}${d.isCenter ? ' center' : ''}`}>
      <Handle type="target" position={Position.Left} />
      <span className="graph-node-name">{d.label}</span>
      <span className="graph-node-type">{d.typeLabel}</span>
      <Handle type="source" position={Position.Right} />
    </div>
  );
}

const nodeTypes = { graphNode: GraphNodeComponent };

// Max evidence-conversation nodes to render around a procedure (keep it legible).
const MAX_EVIDENCE_NODES = 6;

// Build a radial layout of neighbor nodes + edges around a center node.
function radialLayout(
  centerId: string,
  neighbors: Array<{ id: string; data: GraphNodeData; edgeLabel: string }>,
): { nodes: Node[]; edges: Edge[] } {
  const n = Math.max(neighbors.length, 1);
  const nodes: Node[] = neighbors.map((nb, i) => ({
    id: nb.id,
    type: 'graphNode',
    data: nb.data,
    position: {
      x: Math.cos((i * 2 * Math.PI) / n) * 260,
      y: Math.sin((i * 2 * Math.PI) / n) * 260,
    },
  }));
  const edges: Edge[] = neighbors.map((nb, i) => ({
    id: `e-${i}`,
    source: centerId,
    target: nb.id,
    label: nb.edgeLabel,
    animated: true,
  }));
  return { nodes, edges };
}

export function MemoryGraphView() {
  // Selection is mutually exclusive: an entity OR a procedure is centered.
  const [graphEntityId, setGraphEntityId] = useState<string | null>(null);
  const [graphProcedureId, setGraphProcedureId] = useState<string | null>(null);

  const { entities, loading: entitiesLoading } = useMemoryEntities('_all', 1, '');
  const { procedures, loading: proceduresLoading } = useMemoryProcedures('_all', 1);
  const { graph, loading: graphLoading } = useEntityGraph(graphEntityId);

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
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            nodeTypes={nodeTypes}
            fitView
          >
            <Background />
            <Controls />
          </ReactFlow>
        )}
      </div>
    </div>
  );
}
