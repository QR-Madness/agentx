/**
 * UsageMetricsSection — Dashboard "Usage & Cost" hero.
 *
 * Aggregates token / cost / latency from conversation_logs via
 * GET /api/metrics/usage. Renders KPI tiles (incl. month-to-date
 * projection), a dual-axis ComposedChart (tokens + cost over time),
 * a cost-by-model bar chart, and side-by-side leaderboards for
 * Top Models and Per-Agent usage.
 */

import { useMemo, useState } from 'react';
import {
  Activity, Zap, DollarSign, Clock, RefreshCw, BarChart3, TrendingUp,
} from 'lucide-react';
import {
  ResponsiveContainer, BarChart, Bar, ComposedChart, Line,
  XAxis, YAxis, CartesianGrid, Tooltip,
} from 'recharts';
import { useUsageMetrics } from '../../lib/hooks';
import { useAgentProfile } from '../../contexts/AgentProfileContext';
import { formatCost, formatLatency, getModelShortName, formatCompact } from '../../lib/format';
import { Button } from '../ui';
import './UsageMetricsSection.css';

const RANGES = [7, 14, 30] as const;

// Chart colors pull from the active theme so they follow theme switches.
const COLOR_ACCENT = 'var(--accent-primary)';
const COLOR_ACCENT_2 = 'var(--accent-secondary)';
const COLOR_GRID = 'var(--border-default)';
const COLOR_AXIS = 'var(--text-muted)';

function tooltipStyle() {
  return {
    background: 'var(--surface-overlay)',
    border: '1px solid var(--border-default)',
    borderRadius: 8,
    color: 'var(--text-primary)',
    fontSize: 12,
  } as const;
}

/** Compute month-to-date cost and a naive linear projection for the full month. */
function projectMonthlyCost(daily: { date: string; cost_total: number }[]): {
  mtd: number;
  projected: number;
} {
  const now = new Date();
  const ym = `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, '0')}`;
  const inMonth = daily.filter((d) => d.date.startsWith(ym));
  const mtd = inMonth.reduce((s, d) => s + (d.cost_total || 0), 0);
  if (!inMonth.length) return { mtd: 0, projected: 0 };
  // Days elapsed = max day-of-month seen in the data (capped at today).
  const maxDay = Math.max(...inMonth.map((d) => Number(d.date.slice(8, 10))));
  const today = now.getDate();
  const daysElapsed = Math.max(1, Math.min(maxDay, today));
  const daysInMonth = new Date(now.getFullYear(), now.getMonth() + 1, 0).getDate();
  return { mtd, projected: (mtd / daysElapsed) * daysInMonth };
}

export function UsageMetricsSection() {
  const [days, setDays] = useState<number>(14);
  const { usage, loading, error, refresh } = useUsageMetrics(days);
  const { profiles } = useAgentProfile();

  const totals = usage?.totals;
  const currency = totals?.cost_currency || 'USD';
  const offline = usage?.unavailable || !!error;
  const hasData = !!totals && totals.turns > 0;

  // Bar chart: cost per model (fall back to tokens when no cost was computed).
  const costMode = (usage?.by_model || []).some((m) => m.cost_total > 0);
  const byModel = useMemo(
    () => (usage?.by_model || []).map((m) => ({
      name: getModelShortName(m.model),
      cost: m.cost_total,
      tokens: m.tokens_total,
    })),
    [usage?.by_model],
  );

  const dailySeries = useMemo(
    () => (usage?.daily || []).map((d) => ({
      date: d.date,
      label: d.date.slice(5),  // MM-DD
      tokens_total: d.tokens_total,
      cost_total: d.cost_total,
    })),
    [usage?.daily],
  );

  const projection = useMemo(
    () => projectMonthlyCost(usage?.daily || []),
    [usage?.daily],
  );

  // Top models leaderboard rows, with share %.
  const topModels = useMemo(() => {
    const totalCost = totals?.cost_total || 0;
    const totalTokens = totals?.tokens_total || 0;
    const useCost = costMode && totalCost > 0;
    return (usage?.by_model || []).slice(0, 6).map((m, i) => ({
      rank: i + 1,
      name: getModelShortName(m.model),
      turns: m.turns,
      tokens: m.tokens_total,
      cost: m.cost_total,
      share: useCost
        ? (m.cost_total / totalCost) * 100
        : totalTokens > 0 ? (m.tokens_total / totalTokens) * 100 : 0,
    }));
  }, [usage?.by_model, totals, costMode]);

  // Per-agent leaderboard rows, resolving agent_id → display name.
  const byAgent = useMemo(() => {
    return (usage?.by_agent || []).slice(0, 6).map((a, i) => {
      const profile = profiles.find((p) => p.agentId === a.agent_id);
      const display =
        profile?.name
        ?? (a.agent_id === '_default' ? 'Default agent' : a.agent_id);
      return {
        rank: i + 1,
        name: display,
        agentId: a.agent_id,
        turns: a.turns,
        tokens: a.tokens_total,
        cost: a.cost_total,
      };
    });
  }, [usage?.by_agent, profiles]);

  return (
    <div className="usage-section card glass">
      <div className="usage-header">
        <div className="usage-title">
          <BarChart3 size={18} className="usage-title-icon" />
          <span>Usage &amp; Cost</span>
        </div>
        <div className="usage-actions">
          <div className="usage-range" role="group" aria-label="Time range">
            {RANGES.map((r) => (
              <button
                key={r}
                className={`usage-range-btn ${days === r ? 'active' : ''}`}
                onClick={() => setDays(r)}
              >
                {r}d
              </button>
            ))}
          </div>
          <Button variant="ghost" onClick={refresh} aria-label="Refresh usage">
            <RefreshCw size={16} />
          </Button>
        </div>
      </div>

      {loading ? (
        <div className="shimmer usage-placeholder" />
      ) : offline ? (
        <p className="usage-empty">Databases offline — usage unavailable.</p>
      ) : !hasData ? (
        <p className="usage-empty">No assistant turns in the last {days} days yet.</p>
      ) : (
        <>
          <div className="usage-tiles">
            <Tile icon={<Activity size={16} />} label="Turns" value={totals!.turns.toLocaleString()} />
            <Tile
              icon={<Zap size={16} />}
              label="Tokens"
              value={formatCompact(totals!.tokens_total)}
              sub={`${formatCompact(totals!.tokens_input)} in / ${formatCompact(totals!.tokens_output)} out`}
            />
            <Tile
              icon={<DollarSign size={16} />}
              label="Est. cost"
              value={totals!.cost_total > 0 ? `~${formatCost(totals!.cost_total, currency)}` : '—'}
              sub={`last ${days} days`}
            />
            <Tile
              icon={<TrendingUp size={16} />}
              label="Projected month"
              value={projection.projected > 0 ? `~${formatCost(projection.projected, currency)}` : '—'}
              sub={projection.mtd > 0 ? `MTD ${formatCost(projection.mtd, currency)}` : 'no spend yet'}
            />
            <Tile
              icon={<Clock size={16} />}
              label="Avg latency"
              value={totals!.avg_latency_ms > 0 ? formatLatency(totals!.avg_latency_ms) : '—'}
            />
          </div>

          <div className="usage-charts">
            <div className="usage-chart">
              <h4 className="usage-chart-title">{costMode ? 'Cost by model' : 'Tokens by model'}</h4>
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={byModel} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke={COLOR_GRID} vertical={false} />
                  <XAxis dataKey="name" tick={{ fill: COLOR_AXIS, fontSize: 11 }} interval={0} angle={-15} textAnchor="end" height={50} />
                  <YAxis
                    tick={{ fill: COLOR_AXIS, fontSize: 11 }}
                    tickFormatter={(v: number) => (costMode ? formatCost(v, currency) : formatCompact(v))}
                    width={56}
                  />
                  <Tooltip
                    contentStyle={tooltipStyle()}
                    cursor={{ fill: 'var(--surface-hover)' }}
                    formatter={(v) => {
                      const n = Number(v);
                      return costMode ? formatCost(n, currency) : `${n.toLocaleString()} tokens`;
                    }}
                  />
                  <Bar dataKey={costMode ? 'cost' : 'tokens'} fill={COLOR_ACCENT} radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>

            <div className="usage-chart">
              <h4 className="usage-chart-title">Tokens &amp; cost over time</h4>
              <ResponsiveContainer width="100%" height={220}>
                <ComposedChart data={dailySeries} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke={COLOR_GRID} vertical={false} />
                  <XAxis dataKey="label" tick={{ fill: COLOR_AXIS, fontSize: 11 }} />
                  <YAxis
                    yAxisId="tokens"
                    tick={{ fill: COLOR_AXIS, fontSize: 11 }}
                    tickFormatter={(v: number) => formatCompact(v)}
                    width={48}
                  />
                  <YAxis
                    yAxisId="cost"
                    orientation="right"
                    tick={{ fill: COLOR_AXIS, fontSize: 11 }}
                    tickFormatter={(v: number) => formatCost(v, currency)}
                    width={56}
                  />
                  <Tooltip
                    contentStyle={tooltipStyle()}
                    cursor={{ stroke: COLOR_ACCENT_2 }}
                    formatter={(v, name) => {
                      const n = Number(v);
                      if (name === 'cost_total') return [formatCost(n, currency), 'Cost'];
                      return [`${n.toLocaleString()} tokens`, 'Tokens'];
                    }}
                  />
                  <Bar yAxisId="tokens" dataKey="tokens_total" fill={COLOR_ACCENT_2} fillOpacity={0.55} radius={[3, 3, 0, 0]} />
                  <Line yAxisId="cost" type="monotone" dataKey="cost_total" stroke={COLOR_ACCENT} strokeWidth={2} dot={{ r: 2 }} />
                </ComposedChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="usage-tables">
            <LeaderboardTable
              title="Top models"
              headers={['#', 'Model', 'Turns', 'Tokens', 'Cost', 'Share']}
              rows={topModels.map((m) => [
                String(m.rank),
                m.name,
                m.turns.toLocaleString(),
                formatCompact(m.tokens),
                costMode ? formatCost(m.cost, currency) : '—',
                `${m.share.toFixed(1)}%`,
              ])}
              emptyText="No models yet"
            />
            <LeaderboardTable
              title="Per-agent breakdown"
              headers={['#', 'Agent', 'Turns', 'Tokens', 'Cost']}
              rows={byAgent.map((a) => [
                String(a.rank),
                a.name,
                a.turns.toLocaleString(),
                formatCompact(a.tokens),
                a.cost > 0 ? formatCost(a.cost, currency) : '—',
              ])}
              emptyText="No agent activity"
            />
          </div>
        </>
      )}
    </div>
  );
}

function Tile({ icon, label, value, sub }: {
  icon: React.ReactNode;
  label: string;
  value: string;
  sub?: string;
}) {
  return (
    <div className="usage-tile">
      <div className="usage-tile-head">
        <span className="usage-tile-icon">{icon}</span>
        <span className="usage-tile-label">{label}</span>
      </div>
      <div className="usage-tile-value">{value}</div>
      {sub && <div className="usage-tile-sub">{sub}</div>}
    </div>
  );
}

function LeaderboardTable({ title, headers, rows, emptyText }: {
  title: string;
  headers: string[];
  rows: string[][];
  emptyText: string;
}) {
  return (
    <div className="usage-table">
      <h4 className="usage-chart-title">{title}</h4>
      {rows.length === 0 ? (
        <p className="usage-empty-small">{emptyText}</p>
      ) : (
        <table className="usage-leaderboard">
          <thead>
            <tr>
              {headers.map((h, i) => (
                <th key={i} className={i === 1 ? 'col-name' : 'col-num'}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((cells, r) => (
              <tr key={r}>
                {cells.map((c, i) => (
                  <td key={i} className={i === 1 ? 'col-name' : 'col-num'}>{c}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
