#!/usr/bin/env node
/**
 * distill-hero-run.mjs — turn a REAL recorded agent run into the compact timeline
 * the hero console replays. Reads a golden SSE fixture from the client test corpus
 * and distills it to ~8 legible beats + an authentic stat line, written to the
 * committed `src/config/hero-run.json` (the build imports that file, so the docs
 * build never depends on the client tree being present).
 *
 * Re-run after picking a different fixture:  node scripts/distill-hero-run.mjs
 *
 * The displayed timestamps are the run's REAL relative times; the console plays the
 * beats at its own snappy cadence (the real run is ~2 minutes).
 */
import { readFileSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

const SCENARIO = 'work-order-background'; // lead recalls → delegates a work order → teammate web_searches → reports → merge
const fixture = fileURLToPath(
  new URL(`../../client/src/test/fixtures/streams/${SCENARIO}/events.jsonl`, import.meta.url),
);
const out = fileURLToPath(new URL('../src/config/hero-run.json', import.meta.url));

/** Parse one `{ id, sse }` line into { ms, event, data }. */
function parseLine(line) {
  const o = JSON.parse(line);
  if (o._state) return null;
  const ms = Number.parseInt(String(o.id).split('-')[0], 10);
  let event = null;
  let data = null;
  for (const part of String(o.sse ?? '').split('\n')) {
    if (part.startsWith('event: ')) event = part.slice(7).trim();
    else if (part.startsWith('data: ')) data = part.slice(6).trim();
  }
  let parsed = {};
  try {
    parsed = data ? JSON.parse(data) : {};
  } catch {
    /* non-JSON payload — ignore */
  }
  return { ms, event, data: parsed };
}

const lines = readFileSync(fixture, 'utf8').split('\n').filter(Boolean);
const events = lines.map(parseLine).filter(Boolean);
const base = events.find((e) => Number.isFinite(e.ms))?.ms ?? 0;
const rel = (ms) => Math.round(((ms - base) / 1000) * 10) / 10;
const first = (ev) => events.find((e) => e.event === ev);
const count = (ev) => events.filter((e) => e.event === ev).length;

const start = first('start')?.data ?? {};
const mem = first('memory_context')?.data ?? {};
const deleg = first('delegation_start') ?? {};
const tool = first('delegation_tool_call')?.data ?? {};
const report = first('work_order_report') ?? first('delegation_complete') ?? {};
const done = first('done')?.data ?? {};

const facts = Array.isArray(mem.facts) ? mem.facts.length : 0;
const toolCalls = count('delegation_tool_call');
const target = deleg.data?.target_agent_id ?? 'teammate';
const toolName = tool.tool ?? 'tool';
const totalS = done.total_time_ms ? Math.round(done.total_time_ms / 1000) : rel(events.at(-1)?.ms ?? base);
const ctxUsed = done.context_used ? `${(done.context_used / 1000).toFixed(1)}k` : '';
const ctxWin = done.context_window ? `${Math.round(done.context_window / 1000)}k` : '';

const usd = (n) => (typeof n === 'number' ? `$${n < 0.1 ? n.toFixed(4).replace(/0+$/, '') : n.toFixed(2)}` : null);

const model = (done.model ?? start.model ?? '').split('/').pop() ?? 'model';

const heroRun = {
  source: SCENARIO,
  task: 'AI × climate research',
  lead: {
    name: done.agent_name ?? 'lead',
    id: done.agent_id ?? 'lead',
    role: 'lead',
    model,
  },
  team: [{ role: 'researcher', id: target }],
  beats: [
    { t: rel(first('memory_context')?.ms ?? base), actor: 'lead', kind: 'recall', label: 'recall', detail: `${facts} memories · self + global` },
    { t: rel(events.find((e) => e.event === 'status' && e.data?.phase === 'composing')?.ms ?? base), actor: 'lead', kind: 'compose', label: 'compose', detail: ctxUsed && ctxWin ? `context · ${ctxUsed} / ${ctxWin}` : 'assemble context' },
    { t: rel(events.find((e) => e.event === 'status' && e.data?.phase === 'thinking')?.ms ?? base), actor: 'lead', kind: 'think', label: done.thinking_pattern ?? 'reason', detail: 'plan the approach' },
    { t: rel(deleg.ms ?? base), actor: 'lead', kind: 'delegate', label: 'delegate', detail: `→ ${target}`, edge: 'researcher' },
    { t: rel(first('delegation_tool_call')?.ms ?? base), actor: 'researcher', kind: 'tool', label: toolName, detail: `×${toolCalls} queries` },
    { t: rel(first('delegation_tool_result')?.ms ?? base), actor: 'researcher', kind: 'result', label: 'results', detail: `${toolCalls} searches · reading` },
    { t: rel(report.ms ?? base), actor: 'researcher', kind: 'report', label: 'work order', detail: `report → ${done.agent_name ?? 'lead'}`, edge: 'lead' },
    { t: totalS, actor: 'lead', kind: 'merge', label: 'AgentResult', detail: `2 agents · ${toolCalls} tools · ${totalS}s` },
  ],
  stats: {
    model,
    pattern: done.thinking_pattern ?? null,
    tokensIn: done.tokens_input ?? null,
    tokensOut: done.tokens_output ?? null,
    cost: usd(done.cost_estimate),
    provider: done.provider ?? null,
  },
};

writeFileSync(out, JSON.stringify(heroRun, null, 2) + '\n');
console.log(`✓ distilled ${SCENARIO} → src/config/hero-run.json (${heroRun.beats.length} beats, ${totalS}s run)`);
