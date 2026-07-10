/**
 * Connector Catalog — curated, known-good MCP servers the user can add with
 * one click, plus the mapping helpers that turn a catalog entry or an MCP
 * registry search result into a prefilled ServerForm draft.
 *
 * Data-only on purpose: entries are plain objects (no fetches, no logos — a
 * brand initial tile keeps the PWA/Tauri bundles offline-safe), so updating
 * the shelf is editing this file. Every URL below was probed live before
 * inclusion (401/405 from a GET = a real MCP endpoint demanding POST/OAuth).
 */
import type { MCPRegistryResult, MCPServerConfigInput } from './api';

/** Seed for the ServerForm: a suggested name + partial config + guidance. */
export interface ServerDraft {
  name?: string;
  config?: Partial<MCPServerConfigInput>;
  guidance?: { title?: string; note?: string; docsUrl?: string };
}

export type CatalogAuthKind = 'oauth-dcr' | 'oauth-byo-client' | 'api-key' | 'none' | 'local';

/** One field the quick-add dialog asks for — the ONLY thing the user must
 *  provide for a guided connector. `apply` folds the value into the config
 *  draft (entries with no inputs quick-add with zero fields). */
export interface QuickInput {
  key: string;
  label: string;
  placeholder?: string;
  required?: boolean;
  /** Render as a password field (client secrets, API keys). */
  secret?: boolean;
  hint?: string;
  /** Pre-filled value — typically a `${VAR}` env reference so credentials
   *  live in the API server's .env instead of mcp_servers.json. */
  defaultValue?: string;
  apply: (config: Partial<MCPServerConfigInput>, value: string) => Partial<MCPServerConfigInput>;
}

export interface CatalogEntry {
  id: string;
  /** Display title (vendor product name). */
  title: string;
  vendor: string;
  category: 'files' | 'dev' | 'productivity' | 'knowledge' | 'local';
  description: string;
  authKind: CatalogAuthKind;
  /** Suggested server name (becomes the mcp_servers.json key). */
  serverName: string;
  config: Partial<MCPServerConfigInput>;
  /** Quick-add fields (omit for zero-field one-click adds). */
  inputs?: QuickInput[];
  setupNote?: string;
  docsUrl?: string;
  brand: { initial: string; color: string };
}

/** Human labels for the auth-kind badge on catalog tiles. */
export const AUTH_KIND_LABELS: Record<CatalogAuthKind, string> = {
  'oauth-dcr': 'OAuth sign-in',
  'oauth-byo-client': 'OAuth · bring your own app',
  'api-key': 'API key',
  none: 'No auth',
  local: 'Local command',
};

export const CONNECTOR_CATALOG: CatalogEntry[] = [
  {
    id: 'google-drive',
    title: 'Google Drive',
    vendor: 'Google',
    category: 'files',
    description: 'Search, read, and create files in your Drive — Google’s official remote MCP server.',
    authKind: 'oauth-byo-client',
    serverName: 'google-drive',
    config: {
      transport: 'streamable_http',
      url: 'https://drivemcp.googleapis.com/mcp/v1',
      auth: {
        type: 'oauth',
        scope: 'https://www.googleapis.com/auth/drive.readonly https://www.googleapis.com/auth/drive.file',
      },
      timeout: 60,
    },
    inputs: [
      {
        key: 'client_id',
        label: 'Client ID',
        placeholder: '….apps.googleusercontent.com',
        required: true,
        defaultValue: '${GOOGLE_DRIVE_CLIENT_ID}',
        hint: 'The ${VAR} default reads from the API server’s .env (see .env.example) — or paste the raw value.',
        apply: (c, v) => ({ ...c, auth: { ...(c.auth ?? { type: 'oauth' as const }), client_id: v } }),
      },
      {
        key: 'client_secret',
        label: 'Client secret',
        placeholder: 'GOCSPX-… (or ${MY_VAR})',
        required: true,
        secret: true,
        defaultValue: '${GOOGLE_DRIVE_CLIENT_SECRET}',
        hint: 'Keep the ${VAR} reference so the secret lives in the environment, not mcp_servers.json.',
        apply: (c, v) => ({ ...c, auth: { ...(c.auth ?? { type: 'oauth' as const }), client_secret: v } }),
      },
    ],
    setupNote: [
      'Google requires a pre-registered OAuth app (no dynamic registration):',
      '1. In Google Cloud console, create/select a project and enable BOTH the Google Drive API and the Google Drive MCP API. (A missing Drive MCP API does not block sign-in — it surfaces later as every tool call failing with permission errors.)',
      '2. Configure the OAuth consent screen with the two Drive scopes (read-only + drive.file — prefilled).',
      '3. Create an OAuth client ID (Web application) and add this exact redirect URI: http://localhost:12319/api/mcp/oauth/callback (gateway/cluster setups use https://<cluster-host>/api/mcp/oauth/callback — set AGENTX_OAUTH_REDIRECT_URL to match; one Google app can list every cluster’s URI).',
      '4. Set GOOGLE_DRIVE_CLIENT_ID / GOOGLE_DRIVE_CLIENT_SECRET in the API’s .env (defaults below reference them), or paste values directly.',
    ].join('\n'),
    docsUrl: 'https://developers.google.com/workspace/drive/api/guides/configure-mcp-server',
    brand: { initial: 'D', color: '#4285f4' },
  },
  {
    id: 'github',
    title: 'GitHub',
    vendor: 'GitHub',
    category: 'dev',
    description: 'Repos, issues, pull requests, and code search via GitHub’s official remote MCP server.',
    authKind: 'oauth-dcr',
    serverName: 'github',
    config: {
      transport: 'streamable_http',
      url: 'https://api.githubcopilot.com/mcp/',
      auth: { type: 'oauth' },
      timeout: 60,
    },
    docsUrl: 'https://github.com/github/github-mcp-server',
    brand: { initial: 'G', color: '#8250df' },
  },
  {
    id: 'notion',
    title: 'Notion',
    vendor: 'Notion',
    category: 'productivity',
    description: 'Search, read, and write pages and databases in your Notion workspace.',
    authKind: 'oauth-dcr',
    serverName: 'notion',
    config: {
      transport: 'streamable_http',
      url: 'https://mcp.notion.com/mcp',
      auth: { type: 'oauth' },
    },
    docsUrl: 'https://developers.notion.com/docs/mcp',
    brand: { initial: 'N', color: '#9b9b9b' },
  },
  {
    id: 'linear',
    title: 'Linear',
    vendor: 'Linear',
    category: 'productivity',
    description: 'Issues, projects, and cycles — create and update work in Linear.',
    authKind: 'oauth-dcr',
    serverName: 'linear',
    config: {
      transport: 'streamable_http',
      url: 'https://mcp.linear.app/mcp',
      auth: { type: 'oauth' },
    },
    docsUrl: 'https://linear.app/docs/mcp',
    brand: { initial: 'L', color: '#5e6ad2' },
  },
  {
    id: 'sentry',
    title: 'Sentry',
    vendor: 'Sentry',
    category: 'dev',
    description: 'Query issues, events, and traces from your Sentry organization.',
    authKind: 'oauth-dcr',
    serverName: 'sentry',
    config: {
      transport: 'streamable_http',
      url: 'https://mcp.sentry.dev/mcp',
      auth: { type: 'oauth' },
    },
    docsUrl: 'https://docs.sentry.io/product/sentry-mcp/',
    brand: { initial: 'S', color: '#8d5494' },
  },
  {
    id: 'atlassian',
    title: 'Atlassian (Jira & Confluence)',
    vendor: 'Atlassian',
    category: 'productivity',
    description: 'Jira issues and Confluence pages via Atlassian’s official remote MCP server.',
    authKind: 'oauth-dcr',
    serverName: 'atlassian',
    config: {
      transport: 'sse',
      url: 'https://mcp.atlassian.com/v1/sse',
      auth: { type: 'oauth' },
      timeout: 60,
    },
    docsUrl: 'https://support.atlassian.com/rovo/docs/getting-started-with-the-atlassian-remote-mcp-server/',
    brand: { initial: 'A', color: '#1868db' },
  },
  {
    id: 'context7',
    title: 'Context7',
    vendor: 'Upstash',
    category: 'knowledge',
    description: 'Fresh, version-specific library documentation for coding questions.',
    authKind: 'api-key',
    serverName: 'context7',
    config: {
      transport: 'streamable_http',
      url: 'https://mcp.context7.com/mcp',
    },
    inputs: [
      {
        key: 'api_key',
        label: 'API key (optional)',
        placeholder: 'ctx7-… — blank = low rate limit',
        secret: true,
        apply: (c, v) => v.trim()
          ? { ...c, headers: { ...(c.headers ?? {}), CONTEXT7_API_KEY: v.trim() } }
          : c,
      },
    ],
    setupNote: 'Works without a key at a low rate limit. For higher limits, create a free API key at context7.com/dashboard.',
    docsUrl: 'https://github.com/upstash/context7',
    brand: { initial: 'C7', color: '#3ecf8e' },
  },
  {
    id: 'cloudflare-docs',
    title: 'Cloudflare Docs',
    vendor: 'Cloudflare',
    category: 'knowledge',
    description: 'Search Cloudflare’s developer documentation. Open — no sign-in needed.',
    authKind: 'none',
    serverName: 'cloudflare-docs',
    config: {
      transport: 'streamable_http',
      url: 'https://docs.mcp.cloudflare.com/mcp',
    },
    docsUrl: 'https://github.com/cloudflare/mcp-server-cloudflare',
    brand: { initial: 'CF', color: '#f38020' },
  },
  {
    id: 'hugging-face',
    title: 'Hugging Face',
    vendor: 'Hugging Face',
    category: 'knowledge',
    description: 'Search models, datasets, papers, and Spaces on the Hugging Face Hub.',
    authKind: 'none',
    serverName: 'hugging-face',
    config: {
      transport: 'streamable_http',
      url: 'https://huggingface.co/mcp',
    },
    inputs: [
      {
        key: 'hf_token',
        label: 'HF token (optional)',
        placeholder: 'hf_… — blank = anonymous',
        secret: true,
        apply: (c, v) => v.trim()
          ? { ...c, headers: { ...(c.headers ?? {}), Authorization: `Bearer ${v.trim()}` } }
          : c,
      },
    ],
    setupNote: 'Works anonymously; sign-in-gated actions need a Hugging Face token.',
    docsUrl: 'https://huggingface.co/settings/mcp',
    brand: { initial: 'HF', color: '#ffb000' },
  },
  {
    id: 'filesystem',
    title: 'Filesystem',
    vendor: 'Model Context Protocol',
    category: 'local',
    description: 'Read and write files under a directory you choose (runs locally via npx).',
    authKind: 'local',
    serverName: 'filesystem',
    config: {
      transport: 'stdio',
      command: 'npx',
      args: ['-y', '@modelcontextprotocol/server-filesystem'],
    },
    inputs: [
      {
        key: 'directory',
        label: 'Directory to allow',
        placeholder: '/home/you/documents',
        required: true,
        hint: 'The server can read and write ONLY under this path (on the API host).',
        apply: (c, v) => ({ ...c, args: [...(c.args ?? []), v.trim()] }),
      },
    ],
    setupNote: 'Runs locally via npx — requires Node.js on the API host.',
    docsUrl: 'https://github.com/modelcontextprotocol/servers',
    brand: { initial: 'FS', color: '#64748b' },
  },
  {
    id: 'mcp-memory',
    title: 'Knowledge Graph Memory',
    vendor: 'Model Context Protocol',
    category: 'local',
    description: 'A simple local knowledge-graph memory (reference server, runs via npx).',
    authKind: 'local',
    serverName: 'kg-memory',
    config: {
      transport: 'stdio',
      command: 'npx',
      args: ['-y', '@modelcontextprotocol/server-memory'],
    },
    setupNote: 'Requires Node.js on the API host. Note: AgentX has its own agent memory — this is a separate, tool-driven graph.',
    docsUrl: 'https://github.com/modelcontextprotocol/servers',
    brand: { initial: 'KG', color: '#64748b' },
  },
];

/** Categories in display order, with section labels. */
export const CATALOG_CATEGORIES: { id: CatalogEntry['category']; label: string }[] = [
  { id: 'files', label: 'Files & Storage' },
  { id: 'productivity', label: 'Productivity' },
  { id: 'dev', label: 'Development' },
  { id: 'knowledge', label: 'Knowledge' },
  { id: 'local', label: 'Local servers' },
];

/** ServerForm draft for a catalog entry (the "Open full form" escape hatch). */
export function draftFromCatalogEntry(entry: CatalogEntry): ServerDraft {
  return {
    name: entry.serverName,
    config: entry.config,
    guidance: {
      title: `${entry.title} — ${AUTH_KIND_LABELS[entry.authKind]}`,
      note: entry.setupNote,
      docsUrl: entry.docsUrl,
    },
  };
}

/** Fold quick-add field values into the entry's config (missing/blank optional
 *  values are skipped; the caller validates `required` before calling). */
export function applyQuickInputs(
  entry: CatalogEntry,
  values: Record<string, string>,
): Partial<MCPServerConfigInput> {
  let config = entry.config;
  for (const input of entry.inputs ?? []) {
    const value = values[input.key] ?? '';
    if (!value.trim()) continue;
    config = input.apply(config, value);
  }
  return config;
}

/**
 * Is this catalog entry already configured? Remote entries match on URL
 * (exact, trailing-slash-insensitive); local entries match on command + the
 * package argument, so a different allowed directory still counts as added.
 */
export function catalogEntryConfigured(
  entry: CatalogEntry,
  servers: { url?: string | null; command?: string | null; args?: string[] }[],
): boolean {
  const norm = (u: string) => u.replace(/\/+$/, '');
  if (entry.config.url) {
    const target = norm(entry.config.url);
    return servers.some(s => s.url && norm(s.url) === target);
  }
  if (entry.config.command) {
    const pkg = (entry.config.args ?? []).find(a => a.startsWith('@') || !a.startsWith('-'));
    return servers.some(s =>
      s.command === entry.config.command
      && (!pkg || (s.args ?? []).includes(pkg)),
    );
  }
  return false;
}

/* ── MCP registry search → draft ─────────────────────────────── */

/** Registry remote `type` → our TransportType (unknown types default HTTP). */
function transportFromRemoteType(type: string): string {
  return type === 'sse' ? 'sse' : 'streamable_http';
}

/** Short server name from a registry name like `io.github.owner/thing`. */
export function registryShortName(name: string): string {
  const last = name.split('/').pop() || name;
  return last.replace(/[^A-Za-z0-9._-]/g, '-').toLowerCase();
}

/**
 * ServerForm draft for a registry result — remote endpoints win over
 * packages; packages map per ecosystem (npm→npx, pypi→uvx, oci→docker).
 * Registry data is untrusted: this only PREFILLS the form for user review.
 */
export function draftFromRegistryResult(r: MCPRegistryResult): ServerDraft {
  const guidance = {
    title: r.name,
    note: 'Prefilled from the public MCP registry — review before creating, especially commands and URLs.',
    docsUrl: r.repository_url ?? undefined,
  };
  const remote = r.remotes[0];
  if (remote) {
    return {
      name: registryShortName(r.name),
      config: { transport: transportFromRemoteType(remote.type), url: remote.url },
      guidance,
    };
  }
  const pkg = r.packages[0];
  if (pkg) {
    const map: Record<string, { command: string; args: string[] }> = {
      npm: { command: 'npx', args: ['-y', pkg.identifier] },
      pypi: { command: 'uvx', args: [pkg.identifier] },
      oci: { command: 'docker', args: ['run', '-i', '--rm', pkg.identifier] },
    };
    const run = map[pkg.registry_type] ?? { command: pkg.runtime_hint || pkg.identifier, args: [] };
    return {
      name: registryShortName(r.name),
      config: { transport: 'stdio', command: run.command, args: run.args },
      guidance,
    };
  }
  return { name: registryShortName(r.name), config: {}, guidance };
}
