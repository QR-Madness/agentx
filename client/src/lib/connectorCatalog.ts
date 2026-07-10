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
  category: 'files' | 'productivity' | 'design' | 'business' | 'dev' | 'knowledge' | 'local';
  description: string;
  authKind: CatalogAuthKind;
  /** Suggested server name (becomes the mcp_servers.json key). */
  serverName: string;
  config: Partial<MCPServerConfigInput>;
  /** Quick-add fields (omit for zero-field one-click adds). */
  inputs?: QuickInput[];
  setupNote?: string;
  docsUrl?: string;
  /** Gated: the tile renders dimmed with a "Coming soon" badge and can't be
   *  added — the string is the short reason (shown as the tile tooltip).
   *  Power users who DO qualify can still add the server via Add Server. */
  comingSoon?: string;
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
      'IMPORTANT: Google’s Drive MCP server is a Workspace Developer Preview feature — your Google WORKSPACE account + Cloud project must be enrolled (developers.google.com/workspace/preview; consumer/personal Google accounts cannot enroll; approval takes days). Without enrollment every tool call fails "The caller does not have permission" even though sign-in works perfectly.',
      'Google also requires a pre-registered OAuth app (no dynamic registration):',
      '1. In Google Cloud console, create/select a project and enable BOTH the Google Drive API and the Google Drive MCP API. (A missing Drive MCP API does not block sign-in — it surfaces later as every tool call failing with permission errors.)',
      '2. Configure the OAuth consent screen with the two Drive scopes (read-only + drive.file — prefilled).',
      '3. Create an OAuth client ID (Web application) and add this exact redirect URI: http://localhost:12319/api/mcp/oauth/callback (gateway/cluster setups use https://<cluster-host>/api/mcp/oauth/callback — set AGENTX_OAUTH_REDIRECT_URL to match; one Google app can list every cluster’s URI).',
      '4. Set GOOGLE_DRIVE_CLIENT_ID / GOOGLE_DRIVE_CLIENT_SECRET in the API’s .env (defaults below reference them), or paste values directly.',
    ].join('\n'),
    docsUrl: 'https://developers.google.com/workspace/drive/api/guides/configure-mcp-server',
    comingSoon:
      'Google gates ALL its Workspace MCP servers (Drive, Docs, Gmail, …) behind the Workspace '
      + 'Developer Preview Program — enrollment needs a Google Workspace account + Cloud project '
      + '(consumer Google accounts can’t enroll; approval takes days). Live-verified: sign-in works '
      + 'but every tool call fails "The caller does not have permission" until enrolled. If you ARE '
      + 'enrolled, add it manually via Add Server with the settings from the docs link.',
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
    id: 'asana',
    title: 'Asana',
    vendor: 'Asana',
    category: 'productivity',
    description: 'Tasks, projects, and goals — search and update work in Asana.',
    authKind: 'oauth-dcr',
    serverName: 'asana',
    config: {
      transport: 'sse',
      url: 'https://mcp.asana.com/sse',
      auth: { type: 'oauth' },
    },
    docsUrl: 'https://developers.asana.com/docs/using-asanas-mcp-server',
    brand: { initial: 'As', color: '#f06a6a' },
  },
  {
    id: 'monday',
    title: 'monday.com',
    vendor: 'monday.com',
    category: 'productivity',
    description: 'Boards, items, and workflows — read and update your monday.com workspace.',
    authKind: 'oauth-dcr',
    serverName: 'monday',
    config: {
      transport: 'streamable_http',
      url: 'https://mcp.monday.com/mcp',
      auth: { type: 'oauth' },
    },
    docsUrl: 'https://developer.monday.com/apps/docs/mondaycom-mcp',
    brand: { initial: 'M', color: '#6161ff' },
  },
  {
    id: 'zapier',
    title: 'Zapier',
    vendor: 'Zapier',
    category: 'productivity',
    description: 'One connector, thousands of apps — run actions across your Zapier-connected tools.',
    authKind: 'oauth-dcr',
    serverName: 'zapier',
    config: {
      transport: 'streamable_http',
      url: 'https://mcp.zapier.com/api/mcp/mcp',
      auth: { type: 'oauth' },
      timeout: 60,
    },
    docsUrl: 'https://zapier.com/mcp',
    brand: { initial: 'Z', color: '#ff4f00' },
  },
  {
    id: 'figma',
    title: 'Figma',
    vendor: 'Figma',
    category: 'design',
    description: 'Read designs, components, and variables from your Figma files.',
    authKind: 'oauth-dcr',
    serverName: 'figma',
    config: {
      transport: 'streamable_http',
      url: 'https://mcp.figma.com/mcp',
      auth: { type: 'oauth' },
    },
    docsUrl: 'https://developers.figma.com/docs/figma-mcp-server/',
    brand: { initial: 'F', color: '#a259ff' },
  },
  {
    id: 'canva',
    title: 'Canva',
    vendor: 'Canva',
    category: 'design',
    description: 'Create and edit Canva designs, and export them, from conversation.',
    authKind: 'oauth-dcr',
    serverName: 'canva',
    config: {
      transport: 'streamable_http',
      url: 'https://mcp.canva.com/mcp',
      auth: { type: 'oauth' },
    },
    docsUrl: 'https://www.canva.dev/docs/connect/canva-mcp-server-setup/',
    brand: { initial: 'C', color: '#8b3dff' },
  },
  {
    id: 'stripe',
    title: 'Stripe',
    vendor: 'Stripe',
    category: 'business',
    description: 'Customers, payments, subscriptions, and invoices from your Stripe account.',
    authKind: 'oauth-dcr',
    serverName: 'stripe',
    config: {
      transport: 'streamable_http',
      url: 'https://mcp.stripe.com/',
      auth: { type: 'oauth' },
    },
    docsUrl: 'https://docs.stripe.com/mcp',
    brand: { initial: 'S', color: '#635bff' },
  },
  {
    id: 'paypal',
    title: 'PayPal',
    vendor: 'PayPal',
    category: 'business',
    description: 'Invoices, payments, and disputes via PayPal’s official remote MCP server.',
    authKind: 'oauth-dcr',
    serverName: 'paypal',
    config: {
      transport: 'streamable_http',
      url: 'https://mcp.paypal.com/mcp',
      auth: { type: 'oauth' },
    },
    docsUrl: 'https://www.paypal.ai/docs/tools/mcp-quickstart',
    brand: { initial: 'P', color: '#0070ba' },
  },
  {
    id: 'vercel',
    title: 'Vercel',
    vendor: 'Vercel',
    category: 'dev',
    description: 'Projects, deployments, and logs — manage your Vercel apps.',
    authKind: 'oauth-dcr',
    serverName: 'vercel',
    config: {
      transport: 'streamable_http',
      url: 'https://mcp.vercel.com/',
      auth: { type: 'oauth' },
    },
    docsUrl: 'https://vercel.com/docs/mcp/vercel-mcp',
    brand: { initial: 'V', color: '#737373' },
  },
  {
    id: 'cloudflare-browser',
    title: 'Cloudflare Browser Rendering',
    vendor: 'Cloudflare',
    category: 'dev',
    description: 'Fetch pages, take screenshots, and convert web content in a managed headless browser.',
    authKind: 'oauth-dcr',
    serverName: 'cloudflare-browser',
    config: {
      transport: 'streamable_http',
      url: 'https://browser.mcp.cloudflare.com/mcp',
      auth: { type: 'oauth' },
    },
    docsUrl: 'https://github.com/cloudflare/mcp-server-cloudflare',
    brand: { initial: 'CF', color: '#f38020' },
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
    id: 'deepwiki',
    title: 'DeepWiki',
    vendor: 'Cognition',
    category: 'knowledge',
    description: 'Ask questions about any public GitHub repository. Open — no sign-in needed.',
    authKind: 'none',
    serverName: 'deepwiki',
    config: {
      transport: 'streamable_http',
      url: 'https://mcp.deepwiki.com/mcp',
    },
    docsUrl: 'https://docs.devin.ai/work-with-devin/deepwiki-mcp',
    brand: { initial: 'DW', color: '#2563eb' },
  },
  {
    id: 'microsoft-learn',
    title: 'Microsoft Learn',
    vendor: 'Microsoft',
    category: 'knowledge',
    description: 'Search official Microsoft/Azure documentation. Open — no sign-in needed.',
    authKind: 'none',
    serverName: 'microsoft-learn',
    config: {
      transport: 'streamable_http',
      url: 'https://learn.microsoft.com/api/mcp',
    },
    docsUrl: 'https://github.com/MicrosoftDocs/mcp',
    brand: { initial: 'ML', color: '#0078d4' },
  },
  {
    id: 'playwright',
    title: 'Playwright',
    vendor: 'Microsoft',
    category: 'local',
    description: 'Drive a real browser — navigate, click, fill forms, and screenshot (runs locally via npx).',
    authKind: 'local',
    serverName: 'playwright',
    config: {
      transport: 'stdio',
      command: 'npx',
      args: ['-y', '@playwright/mcp@latest'],
    },
    setupNote: 'Runs locally via npx — requires Node.js on the API host; first run downloads a browser.',
    docsUrl: 'https://github.com/microsoft/playwright-mcp',
    brand: { initial: 'PW', color: '#2ead33' },
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
  { id: 'productivity', label: 'Productivity' },
  { id: 'dev', label: 'Development' },
  { id: 'knowledge', label: 'Knowledge' },
  { id: 'design', label: 'Design' },
  { id: 'business', label: 'Business & Payments' },
  { id: 'files', label: 'Files & Storage' },
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
 * The already-configured server backing this catalog entry, or undefined.
 * Remote entries match on URL (exact, trailing-slash-insensitive); local
 * entries match on command + the package argument, so a different allowed
 * directory still counts as added. Generic over the caller's server shape so
 * the connector dialog gets the full object back (name, status, auth state).
 */
export function findConfiguredServer<
  S extends { url?: string | null; command?: string | null; args?: string[] },
>(entry: CatalogEntry, servers: S[]): S | undefined {
  const norm = (u: string) => u.replace(/\/+$/, '');
  if (entry.config.url) {
    const target = norm(entry.config.url);
    return servers.find(s => s.url && norm(s.url) === target);
  }
  if (entry.config.command) {
    const pkg = (entry.config.args ?? []).find(a => a.startsWith('@') || !a.startsWith('-'));
    return servers.find(s =>
      s.command === entry.config.command
      && (!pkg || (s.args ?? []).includes(pkg)),
    );
  }
  return undefined;
}

/** Boolean convenience over `findConfiguredServer`. */
export function catalogEntryConfigured(
  entry: CatalogEntry,
  servers: { url?: string | null; command?: string | null; args?: string[] }[],
): boolean {
  return findConfiguredServer(entry, servers) !== undefined;
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
