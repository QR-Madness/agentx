// ─── Graph view ────────────────────────────────────────────────────────────

import { useState, useEffect } from 'react';
import { RefreshCw, GitBranch } from 'lucide-react';
import {
  ReactFlow, Background, Controls, Handle, Position,
  useNodesState, useEdgesState,
  type NodeProps, type Node, type Edge
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import { useMemoryEntities, useEntityGraph } from '../../lib/hooks';

// Custom entity node (module-level to avoid re-creation on each render).
type EntityNodeData = { label: string; typeLabel: string; isCenter: boolean } & Record<string, unknown>;

function EntityNodeComponent({ data }: NodeProps) {
  const d = data as EntityNodeData;
  return (
    <div className={`graph-node${d.isCenter ? ' center' : ''}`}>
      <Handle type="target" position={Position.Left} />
      <span className="graph-node-name">{d.label}</span>
      <span className="graph-node-type">{d.typeLabel}</span>
      <Handle type="source" position={Position.Right} />
    </div>
  );
}

const nodeTypes = { entityNode: EntityNodeComponent };

export function MemoryGraphView() {
  const [graphEntityId, setGraphEntityId] = useState<string | null>(null);
  const { entities, loading: entitiesLoading } = useMemoryEntities('_all', 1, '');
  const { graph, loading: graphLoading } = useEntityGraph(graphEntityId);
  const [nodes, setNodes, onNodesChange] = useNodesState<Node>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);

  useEffect(() => {
    if (!graph) {
      setNodes([]);
      setEdges([]);
      return;
    }
    const n = Math.max(graph.relationships.length, 1);
    const center: Node = {
      id: graph.entity.id,
      type: 'entityNode',
      data: { label: graph.entity.name, typeLabel: graph.entity.type, isCenter: true },
      position: { x: 0, y: 0 },
    };
    const neighbors: Node[] = graph.relationships.map((rel, i) => ({
      id: rel.target.id,
      type: 'entityNode',
      data: { label: rel.target.name, typeLabel: rel.target.type, isCenter: false },
      position: {
        x: Math.cos((i * 2 * Math.PI) / n) * 260,
        y: Math.sin((i * 2 * Math.PI) / n) * 260,
      },
    }));
    const newEdges: Edge[] = graph.relationships.map((rel, i) => ({
      id: `e-${i}`,
      source: graph.entity.id,
      target: rel.target.id,
      label: rel.type,
      animated: true,
    }));
    setNodes([center, ...neighbors]);
    setEdges(newEdges);
  }, [graph]);

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
            onClick={() => setGraphEntityId(graphEntityId === e.id ? null : e.id)}
          >
            <span className="graph-btn-name">{e.name}</span>
            <span className="graph-btn-type">{e.type}</span>
          </button>
        ))}
      </div>
      <div className="graph-canvas">
        {!graphEntityId ? (
          <div className="graph-empty">
            <GitBranch size={32} />
            <p>Select an entity to explore its relationships</p>
          </div>
        ) : graphLoading ? (
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
