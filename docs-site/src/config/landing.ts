// Landing-page copy + data. Single source for the marketing surface; kept separate
// from nav.ts (navigation structure) but reusing the same --c-* accent tokens.
// Ported from the design handoff (landing.jsx / common.jsx / system-map.jsx).

/** Product version — keep in sync with ../../../versions.yaml (api.version). */
export const version = '0.21.148';

export type Feature = {
  key: string;
  label: string;
  desc: string;
  color: string; // a --c-* token, referenced via var()
  tag: string;
  href: string; // where the rail card / chip links in the docs
};

// The eight runtime subsystems. Used by the subsystems rail, the system-map copy
// chips, and the constellation. `key` drives the constellation node order.
export const FEATURES: Feature[] = [
  {
    key: 'agent',
    label: 'Agent',
    desc: 'Per-request orchestrator. Sessions, context budgeting, tool loop, output parsing.',
    color: 'var(--c-agent)',
    tag: 'agent/core.py',
    href: '/docs/architecture/overview',
  },
  {
    key: 'reasoning',
    label: 'Reasoning',
    desc: 'Chain-of-Thought · Tree-of-Thought · ReAct · Reflection. Auto-selected per task.',
    color: 'var(--c-reasoning)',
    tag: '4 strategies',
    href: '/docs/features/reasoning',
  },
  {
    key: 'drafting',
    label: 'Drafting',
    desc: 'Speculative decoding, multi-stage pipelines, N-best candidates.',
    color: 'var(--c-drafting)',
    tag: 'spec · pipe · cand',
    href: '/docs/features/drafting',
  },
  {
    key: 'mcp',
    label: 'MCP Client',
    desc: 'Connect to external tool servers over stdio, SSE, or streamable HTTP.',
    color: 'var(--c-mcp)',
    tag: 'stdio · sse · http',
    href: '/docs/features/mcp',
  },
  {
    key: 'providers',
    label: 'Providers',
    desc: 'One interface across LM Studio, Anthropic, OpenAI, OpenRouter, and Vercel AI Gateway — swap models per request.',
    color: 'var(--c-providers)',
    tag: '5 providers · byo models',
    href: '/docs/features/providers',
  },
  {
    key: 'prompts',
    label: 'Prompts',
    desc: 'Profile-based composition with a global prompt layer. Sections compose at runtime.',
    color: 'var(--c-prompts)',
    tag: 'profiles · sections',
    href: '/docs/features/prompts',
  },
  {
    key: 'memory',
    label: 'Memory',
    desc: 'Four memory types with five recall techniques, per-agent self-knowledge, and background consolidation.',
    color: 'var(--c-memory)',
    tag: '4-type · 5 recall',
    href: '/docs/features/memory',
  },
  {
    key: 'translation',
    label: 'Translation',
    desc: 'Two-level language detection + NLLB-200. 200+ languages, no round trip.',
    color: 'var(--c-translation)',
    tag: 'NLLB-200 · 200+ langs',
    href: '/docs/features/translation',
  },
];

// ── Hero: the Alloy multi-agent trace ──────────────────────────────────────
export type AlloyAgent = {
  role: string;
  id: string;
  profile: string;
  model: string;
  color: string;
};

export const ALLOY: AlloyAgent[] = [
  { role: 'planner', id: 'swift-lucid-mole', profile: 'planner v1.2', model: 'claude-3.7-sonnet', color: 'var(--c-agent)' },
  { role: 'researcher', id: 'quiet-amber-fox', profile: 'researcher v0.6  extends default', model: 'claude-3.7-sonnet', color: 'var(--c-memory)' },
  { role: 'writer', id: 'brisk-cobalt-otter', profile: 'writer v0.3  extends default', model: 'gpt-4o', color: 'var(--c-drafting)' },
];

export type TraceLine = { t: string; r: string; k: string; l: string; m: string };

export const TRACE: TraceLine[] = [
  { t: '+000', r: 'planner', k: 'req', l: 'POST /agent/run', m: 'task: "draft Q2 roadmap brief from open issues"' },
  { t: '+008', r: 'planner', k: 'profile', l: 'PromptManager', m: 'compose · planner v1.2 · 4 sections (persona · task · constraints · format)' },
  { t: '+019', r: 'planner', k: 'memory', l: 'MemoryBundle', m: 'recall · _self_swift-lucid-mole + _global → 4 procedural, 2 semantic' },
  { t: '+031', r: 'planner', k: 'plan', l: 'TaskPlanner', m: 'decompose → 3 subtasks · strategy: ReAct' },
  { t: '+044', r: 'planner', k: 'alloy', l: 'Alloy.spawn  ×2', m: 'delegate plan · researcher · writer' },
  { t: '+052', r: 'researcher', k: 'profile', l: 'PromptManager', m: 'compose · researcher v0.6  extends default · sections [+2/-1]' },
  { t: '+061', r: 'researcher', k: 'memory', l: 'MemoryBundle', m: 'recall · _self_quiet-amber-fox + _global → 6 facts · hybrid + HyDE' },
  { t: '+089', r: 'researcher', k: 'tool', l: 'mcp.gh.list_issues', m: "{ repo: 'agentx', state: 'open', label: 'phase-16' } → 14 issues" },
  { t: '+331', r: 'researcher', k: 'tool', l: 'mcp.gh.read_issue', m: '#142 #149 #157 · 41 KB total' },
  { t: '+498', r: 'researcher', k: 'reason', l: 'Reasoning.ReAct', m: 'thought × 8 · action × 3 · select 5 issues · summarize' },
  { t: '+712', r: 'researcher', k: 'memory', l: 'MemoryWrite', m: "episodic + semantic · entities: ['#142', '#149', 'plan-execution']" },
  { t: '+730', r: 'researcher', k: 'return', l: 'SubResult → planner', m: 'compressed 41 KB → 4.2 KB · salience: high' },
  { t: '+744', r: 'writer', k: 'profile', l: 'PromptManager', m: "compose · writer v0.3  extends default · constraints: 'concise · cite'" },
  { t: '+758', r: 'writer', k: 'memory', l: 'MemoryBundle', m: 'recall · _self_brisk-cobalt-otter → 3 procedural (tone, format)' },
  { t: '+784', r: 'writer', k: 'draft', l: 'Drafting.pipeline', m: 'outline → expand → review · 3 stages · candidate top score 0.91' },
  { t: '+1.2', r: 'writer', k: 'return', l: 'SubResult → planner', m: 'brief.md · 1,842 tokens · 4 cited issues' },
  { t: '+1.3', r: 'planner', k: 'merge', l: 'AgentResult', m: '3 agents · 4 tool calls · 4,621 tokens · 1,308 ms' },
];

// ── Hero stat row ───────────────────────────────────────────────────────────
export const HERO_STATS: { l: string; v: string }[] = [
  { l: 'deploy', v: 'Self-hosted · air-gappable' },
  { l: 'backend', v: 'Django 5.2' },
  { l: 'client', v: 'Tauri v2 · React 19' },
  { l: 'memory', v: 'Neo4j · pgvector · Redis' },
  { l: 'license', v: 'MIT' },
];

// ── Lifecycle band ────────────────────────────────────────────────────────
export type LifecycleStep = { c: string; k: string; l: string; t: string };

export const LIFECYCLE: LifecycleStep[] = [
  { c: 'var(--c-agent)', k: '01', l: 'views.py', t: 'dispatch · session bind' },
  { c: 'var(--c-prompts)', k: '02', l: 'PromptManager', t: 'compose profile + sections (extends)' },
  { c: 'var(--c-memory)', k: '03', l: 'MemoryBundle', t: 'recall · _self_{agent} + _global · hybrid + HyDE' },
  { c: 'var(--c-reasoning)', k: '04', l: 'Reasoning', t: 'select strategy (CoT/ToT/ReAct/Reflection)' },
  { c: 'var(--c-agent)', k: '05', l: 'Alloy.spawn', t: 'TaskPlanner delegates plan to N sub-agents' },
  { c: 'var(--c-mcp)', k: '06', l: 'ToolExecutor', t: 'stdio · sse · http  →  external MCP' },
  { c: 'var(--c-drafting)', k: '07', l: 'Drafting', t: 'candidate / pipeline / speculative' },
  { c: 'var(--color-ok)', k: '08', l: 'AgentResult', t: 'merge sub-results · parse · persist · respond' },
];

// ── Data layer ────────────────────────────────────────────────────────────
export type Store = { c: string; n: string; d: string };

export const STORES: Store[] = [
  { c: 'var(--c-memory)', n: 'Neo4j 5.15', d: 'entity graphs · semantic + procedural' },
  { c: 'var(--c-agent)', n: 'PostgreSQL + pgvector', d: 'vectors · episodic · audit log' },
  { c: 'var(--c-drafting)', n: 'Redis 7', d: 'working memory cache · session state' },
];

// ── "Why AgentX" — the differentiators band ─────────────────────────────────
export type Differentiator = {
  title: string;
  desc: string;
  color: string; // a --c-* token
  tag: string;
  href: string;
};

export const DIFFERENTIATORS: Differentiator[] = [
  {
    title: 'Glassbox, not a black box',
    desc: 'Watch every decision the loop makes — profile compose, memory recall, reasoning, delegation, and each tool call stream live. Nothing happens off-screen.',
    color: 'var(--c-reasoning)',
    tag: 'live trace',
    href: '/docs/architecture/overview',
  },
  {
    title: 'Memory that actually learns',
    desc: 'Episodic, semantic, procedural, and working memory — with five recall techniques, per-agent self-knowledge channels, and background consolidation. Agents remember across conversations.',
    color: 'var(--c-memory)',
    tag: '4 types · 5 recall',
    href: '/docs/features/memory',
  },
  {
    title: 'Agent Alloy — multi-agent',
    desc: 'A supervisor delegates to specialist agents over a shared memory channel. @-mention routing, ad-hoc agent-to-agent delegation, and per-agent tool isolation.',
    color: 'var(--c-agent)',
    tag: 'supervisor · specialists',
    href: '/docs/features/multi-agent',
  },
  {
    title: 'Self-hosted · bring your own models',
    desc: 'Run your own server, air-gappable over DNS — no SaaS lock-in. One interface across LM Studio, Anthropic, OpenAI, OpenRouter, and Vercel AI Gateway.',
    color: 'var(--c-providers)',
    tag: 'you run it',
    href: '/docs/features/providers',
  },
  {
    title: 'Runs where you do',
    desc: 'One Tauri v2 client across Windows, macOS, and Linux — plus an Android build (mobile-ready) and a browser/web mode. Same app, every platform.',
    color: 'var(--color-accent)',
    tag: 'desktop · mobile · web',
    href: '/docs/getting-started/installation',
  },
  {
    title: 'One-line internal tools',
    desc: 'Turn any Python function into an agent tool with a single @register_tool decorator — no MCP server required for in-process tools.',
    color: 'var(--c-mcp)',
    tag: '@register_tool',
    href: '/docs/features/mcp',
  },
  {
    title: 'Context that never overflows',
    desc: 'Oversized tool outputs are compressed and stored for retrieval; Focus-style trajectory compression keeps long tool loops inside the window.',
    color: 'var(--c-drafting)',
    tag: 'context gating',
    href: '/docs/architecture/overview',
  },
  {
    title: 'Ambassador — a parallel narrator',
    desc: 'A dedicated agent runs alongside your conversation and briefs you on it — by voice or text — without ever entering or polluting the transcript. Come back and ask what your agents have been doing.',
    color: 'var(--c-translation)',
    tag: 'parallel relay · voice',
    href: '/docs/features/multi-agent',
  },
  {
    title: 'Composable prompts + enhancer',
    desc: 'Profile-based prompt composition over a global layer, reusable templates, and an LLM-backed prompt enhancer that rewrites prompts on demand.',
    color: 'var(--c-prompts)',
    tag: 'profiles · enhancer',
    href: '/docs/features/prompts',
  },
];

// ── "Direction" — roadmap teaser ────────────────────────────────────────────
export type DirectionColumn = { k: string; title: string; items: string[] };

export const DIRECTION: DirectionColumn[] = [
  {
    k: 'shipped',
    title: 'Shipped',
    items: [
      'Mobile-Ready Alpha — self-hosted server (auth · prod Docker · clusters) + Tauri v2 desktop/Android',
      'Layered memory — 4 types, 5-technique recall, per-agent self-knowledge, background consolidation',
      'Agent Alloy — supervisor + specialist delegation, @-mention routing, per-agent tool isolation',
      'Ambassador — a parallel briefer with two-way voice (TTS + STT)',
    ],
  },
  {
    k: 'now',
    title: 'Shipping now',
    items: [
      'Ambassador v2 — a conversational, tool-using relay with its own persistent thread',
      'Procedural memory — agents distill and replay "how we work here"',
      'Active recall + memory tuning — closing the query-formulation gap',
    ],
  },
  {
    k: 'next',
    title: 'Next',
    items: [
      'Memory hardening — event-sourced write path, bitemporal facts, a capability registry',
      'File Workspaces & Document RAG — hand agents a corpus, not just memory + web',
      'Settings Advisor + Manifest — a self-tuning control plane',
      'Factory canvas — a visual multi-agent workflow editor',
    ],
  },
];

// ── "Powered by" — open-source shout-out ────────────────────────────────────
export type Dependency = { name: string; href: string };

export const POWERED_BY: Dependency[] = [
  { name: 'Django', href: 'https://www.djangoproject.com/' },
  { name: 'Tauri', href: 'https://tauri.app/' },
  { name: 'React', href: 'https://react.dev/' },
  { name: 'Vite', href: 'https://vite.dev/' },
  { name: 'Tailwind CSS', href: 'https://tailwindcss.com/' },
  { name: 'NLLB-200', href: 'https://huggingface.co/facebook/nllb-200-distilled-600M' },
  { name: 'Hugging Face', href: 'https://huggingface.co/' },
  { name: 'sentence-transformers', href: 'https://www.sbert.net/' },
  { name: 'Neo4j', href: 'https://neo4j.com/' },
  { name: 'PostgreSQL', href: 'https://www.postgresql.org/' },
  { name: 'pgvector', href: 'https://github.com/pgvector/pgvector' },
  { name: 'Redis', href: 'https://redis.io/' },
  { name: 'Model Context Protocol', href: 'https://modelcontextprotocol.io/' },
];
