/**
 * SystemStatusStrip — Compact health summary for the Dashboard.
 *
 * Collapses the eight legacy status cards into one slim glass strip of
 * color-coded pills (API · Neo4j · PG · Redis · Providers · MCP · Agent),
 * with a "Details" toggle that expands the full subsystem breakdown
 * inline. Reuses every hook the old cards consumed — no data layer
 * changes.
 */

import { useState } from 'react';
import {
  Activity, Database, Brain, Wrench, Layers, HardDrive,
  MessageSquare, ChevronDown, RefreshCw,
} from 'lucide-react';
import {
  useHealth, useProviders, useMCPServers, useAgentStatus, useMemoryStats,
} from '../../lib/hooks';
import { useConversation } from '../../contexts/ConversationContext';
import { Button } from '../ui';
import './SystemStatusStrip.css';

type Tone = 'online' | 'warning' | 'offline' | 'loading';

function pillTone(status?: string | null, loading?: boolean): Tone {
  if (loading) return 'loading';
  if (!status) return 'offline';
  if (status === 'healthy' || status === 'ready' || status === 'connected') return 'online';
  if (status === 'degraded') return 'warning';
  return 'offline';
}

function formatBytes(mb: number | string | undefined | null): string {
  if (mb === undefined || mb === null) return '—';
  const v = typeof mb === 'string' ? parseFloat(mb) : mb;
  if (isNaN(v)) return '—';
  if (v < 1) return `${(v * 1024).toFixed(0)} KB`;
  if (v < 1024) return `${v.toFixed(1)} MB`;
  return `${(v / 1024).toFixed(2)} GB`;
}

export function SystemStatusStrip() {
  const { data: health, loading: healthLoading, refresh: refreshHealth } = useHealth();
  const { providers, loading: providersLoading } = useProviders();
  const { servers: mcpServers, loading: mcpLoading } = useMCPServers();
  const { status: agentStatus } = useAgentStatus();
  const { stats: memoryStats, loading: memoryLoading } = useMemoryStats();
  const { serverConversations } = useConversation();

  const [expanded, setExpanded] = useState(false);

  const providersOnline = providers.filter((p) => p.available).length;
  const mcpOnline = mcpServers.filter((s) => s.status === 'connected').length;

  const pills: Array<{ label: string; tone: Tone; sub?: string }> = [
    { label: 'API', tone: pillTone(health?.api?.status, healthLoading) },
    { label: 'Neo4j', tone: pillTone(health?.memory?.neo4j?.status, healthLoading) },
    { label: 'PG', tone: pillTone(health?.memory?.postgres?.status, healthLoading) },
    { label: 'Redis', tone: pillTone(health?.memory?.redis?.status, healthLoading) },
    {
      label: 'Providers',
      tone: providersLoading ? 'loading' : providersOnline > 0 ? 'online' : 'offline',
      sub: providersLoading ? '…' : `${providersOnline}/${providers.length}`,
    },
    {
      label: 'MCP',
      tone: mcpLoading ? 'loading' : mcpServers.length === 0 ? 'offline' : mcpOnline > 0 ? 'online' : 'warning',
      sub: mcpLoading ? '…' : `${mcpServers.length}`,
    },
    {
      label: 'Agent',
      tone: pillTone(agentStatus?.status),
      sub: agentStatus?.status,
    },
  ];

  return (
    <div className={`status-strip card glass ${expanded ? 'expanded' : ''}`}>
      <div className="strip-row">
        <div className="strip-pills">
          {pills.map((p) => (
            <span key={p.label} className={`strip-pill tone-${p.tone}`} title={p.sub ? `${p.label}: ${p.sub}` : p.label}>
              <span className="strip-dot" />
              <span className="strip-pill-label">{p.label}</span>
              {p.sub && <span className="strip-pill-sub">{p.sub}</span>}
            </span>
          ))}
        </div>
        <div className="strip-actions">
          <Button variant="ghost" onClick={refreshHealth} aria-label="Refresh">
            <RefreshCw size={14} />
          </Button>
          <button
            type="button"
            className="strip-toggle"
            onClick={() => setExpanded((v) => !v)}
            aria-expanded={expanded}
          >
            Details
            <ChevronDown size={14} className={`strip-chevron ${expanded ? 'open' : ''}`} />
          </button>
        </div>
      </div>

      {/* Expandable detail panel — uses grid-template-rows trick for animatable height. */}
      <div className="strip-detail-wrap">
        <div className="strip-detail">
          <div className="strip-detail-grid">
            <DetailGroup icon={<Activity size={14} />} title="System">
              <Row k="API" v={health?.api?.status ?? 'offline'} />
              <Row k="Translation" v={health?.translation?.status ?? 'offline'} />
              {health?.version && <Row k="Version" v={health.version} />}
            </DetailGroup>

            <DetailGroup icon={<Database size={14} />} title="Databases">
              <Row k="Neo4j" v={health?.memory?.neo4j?.status ?? 'offline'} />
              <Row k="PostgreSQL" v={health?.memory?.postgres?.status ?? 'offline'} />
              <Row k="Redis" v={health?.memory?.redis?.status ?? 'offline'} />
            </DetailGroup>

            <DetailGroup icon={<Brain size={14} />} title="Providers" count={`${providersOnline}/${providers.length}`}>
              {providersLoading ? (
                <p className="strip-empty">…</p>
              ) : providers.length === 0 ? (
                <p className="strip-empty">None configured</p>
              ) : (
                providers.slice(0, 4).map((p) => (
                  <Row key={p.name} k={p.name} v={p.available ? 'available' : 'offline'} />
                ))
              )}
            </DetailGroup>

            <DetailGroup icon={<Wrench size={14} />} title="MCP" count={`${mcpServers.length}`}>
              {mcpLoading ? (
                <p className="strip-empty">…</p>
              ) : mcpServers.length === 0 ? (
                <p className="strip-empty">No servers</p>
              ) : (
                mcpServers.slice(0, 4).map((s) => (
                  <Row key={s.name} k={s.name} v={s.status ?? 'offline'} />
                ))
              )}
            </DetailGroup>

            <DetailGroup icon={<Layers size={14} />} title="Memory">
              {memoryLoading ? (
                <p className="strip-empty">…</p>
              ) : !memoryStats || memoryStats.unavailable ? (
                <p className="strip-empty">offline</p>
              ) : (
                <>
                  <Row k="Entities" v={memoryStats.totals.entities.toLocaleString()} />
                  <Row k="Facts" v={memoryStats.totals.facts.toLocaleString()} />
                  <Row k="Turns" v={memoryStats.totals.turns.toLocaleString()} />
                </>
              )}
            </DetailGroup>

            <DetailGroup icon={<HardDrive size={14} />} title="Storage">
              <Row k="PostgreSQL" v={formatBytes(health?.storage?.postgres_size_mb)} />
              <Row k="Neo4j" v={formatBytes(health?.storage?.neo4j_size_mb)} />
              <Row k="Redis" v={formatBytes(health?.storage?.redis_memory_mb)} />
            </DetailGroup>

            <DetailGroup icon={<MessageSquare size={14} />} title="Conversations" count={`${serverConversations.length}`}>
              <Row k="Saved" v={serverConversations.length.toLocaleString()} />
              <Row
                k="Channels"
                v={new Set(serverConversations.map((c) => c.channel)).size.toLocaleString()}
              />
            </DetailGroup>

            <DetailGroup icon={<Activity size={14} />} title="Agent">
              <Row k="Status" v={agentStatus?.status ?? '—'} />
              <Row k="Active sessions" v={(agentStatus?.active_sessions ?? 0).toLocaleString()} />
            </DetailGroup>
          </div>
        </div>
      </div>
    </div>
  );
}

function DetailGroup({
  icon, title, count, children,
}: { icon: React.ReactNode; title: string; count?: string; children: React.ReactNode }) {
  return (
    <div className="strip-detail-group">
      <div className="strip-detail-head">
        <span className="strip-detail-icon">{icon}</span>
        <span className="strip-detail-title">{title}</span>
        {count && <span className="strip-detail-count">{count}</span>}
      </div>
      <div className="strip-detail-body">{children}</div>
    </div>
  );
}

function Row({ k, v }: { k: string; v: string | number }) {
  const tone = typeof v === 'string'
    ? (['healthy', 'available', 'connected', 'ready', 'online'].includes(v) ? 'online'
      : ['degraded', 'warning'].includes(v) ? 'warning'
      : ['offline', 'unhealthy', 'disconnected'].includes(v) ? 'offline'
      : 'neutral')
    : 'neutral';
  return (
    <div className="strip-row-kv">
      <span className="strip-row-k">{k}</span>
      <span className={`strip-row-v tone-${tone}`}>{v}</span>
    </div>
  );
}
