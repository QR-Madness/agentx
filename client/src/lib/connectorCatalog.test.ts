import { describe, it, expect } from 'vitest';
import type { MCPRegistryResult } from './api';
import {
  CONNECTOR_CATALOG, CATALOG_CATEGORIES, LENSES, AUTH_BADGE, applyQuickInputs,
  catalogEntryConfigured, findConfiguredServer, draftFromCatalogEntry,
  draftFromRegistryResult, registryShortName,
} from './connectorCatalog';

describe('CONNECTOR_CATALOG entries', () => {
  it('every entry yields a form-fillable draft (remote url or local command)', () => {
    for (const entry of CONNECTOR_CATALOG) {
      const draft = draftFromCatalogEntry(entry);
      expect(draft.name, entry.id).toBeTruthy();
      const cfg = draft.config!;
      if (cfg.transport === 'stdio') {
        expect(cfg.command, entry.id).toBeTruthy();
      } else {
        expect(cfg.url, entry.id).toMatch(/^https:\/\//);
      }
    }
  });

  it('OAuth entries carry an oauth auth block; byo-client entries explain setup', () => {
    for (const entry of CONNECTOR_CATALOG) {
      if (entry.authKind === 'oauth-dcr' || entry.authKind === 'oauth-byo-client') {
        expect(entry.config.auth?.type, entry.id).toBe('oauth');
      }
      if (entry.authKind === 'oauth-byo-client') {
        expect(entry.setupNote, entry.id).toBeTruthy();
      }
    }
  });

  it('ids and suggested server names are unique', () => {
    const ids = CONNECTOR_CATALOG.map(e => e.id);
    const names = CONNECTOR_CATALOG.map(e => e.serverName);
    expect(new Set(ids).size).toBe(ids.length);
    expect(new Set(names).size).toBe(names.length);
  });

  it('every entry sits in a declared category section', () => {
    const known = new Set(CATALOG_CATEGORIES.map(c => c.id));
    for (const entry of CONNECTOR_CATALOG) {
      expect(known.has(entry.category), `${entry.id} → ${entry.category}`).toBe(true);
    }
  });

  it('every auth kind has a tile badge label', () => {
    for (const entry of CONNECTOR_CATALOG) {
      expect(AUTH_BADGE[entry.authKind], entry.id).toBeTruthy();
    }
  });

  it('gates Google Drive behind Coming soon (Workspace preview only)', () => {
    const gdrive = CONNECTOR_CATALOG.find(e => e.id === 'google-drive')!;
    expect(gdrive.comingSoon).toBeTruthy();
    expect(gdrive.comingSoon).toMatch(/preview/i);
  });
});

describe('LENSES partition', () => {
  it('every category belongs to exactly one lens', () => {
    const seen = new Map<string, number>();
    for (const lens of LENSES) {
      for (const cat of lens.categoryIds) seen.set(cat, (seen.get(cat) ?? 0) + 1);
    }
    for (const cat of CATALOG_CATEGORIES) {
      expect(seen.get(cat.id), `${cat.id} lens count`).toBe(1);
    }
  });

  it("every lens category is a real category, and every entry's category is claimed by a lens", () => {
    const catIds = new Set(CATALOG_CATEGORIES.map(c => c.id));
    const claimed = new Set(LENSES.flatMap(l => l.categoryIds));
    for (const lens of LENSES) {
      for (const cat of lens.categoryIds) {
        expect(catIds.has(cat), `${lens.id} → ${cat}`).toBe(true);
      }
    }
    for (const entry of CONNECTOR_CATALOG) {
      expect(claimed.has(entry.category), `${entry.id} → ${entry.category} unclaimed`).toBe(true);
    }
  });

  it('leads with Global Intelligence', () => {
    expect(LENSES[0].id).toBe('global');
  });
});

describe('applyQuickInputs', () => {
  const gdrive = CONNECTOR_CATALOG.find(e => e.id === 'google-drive')!;
  const fs = CONNECTOR_CATALOG.find(e => e.id === 'filesystem')!;
  const context7 = CONNECTOR_CATALOG.find(e => e.id === 'context7')!;

  it('folds Google Drive credentials into the oauth block, keeping the scope', () => {
    const cfg = applyQuickInputs(gdrive, { client_id: 'cid.apps', client_secret: 'shh' });
    expect(cfg.auth).toMatchObject({ type: 'oauth', client_id: 'cid.apps', client_secret: 'shh' });
    expect(cfg.auth?.scope).toContain('drive.readonly');
    // The base entry is not mutated.
    expect(gdrive.config.auth?.client_id).toBeUndefined();
  });

  it('appends the filesystem directory as the final arg', () => {
    const cfg = applyQuickInputs(fs, { directory: '/home/me/docs' });
    expect(cfg.args?.at(-1)).toBe('/home/me/docs');
    expect(fs.config.args).not.toContain('/home/me/docs');
  });

  it('skips blank optional values (Context7 works keyless)', () => {
    expect(applyQuickInputs(context7, { api_key: '  ' }).headers).toBeUndefined();
    expect(applyQuickInputs(context7, { api_key: 'k' }).headers).toMatchObject({ CONTEXT7_API_KEY: 'k' });
  });

  it('Google Drive defaults reference the .env vars (the .env.example contract)', () => {
    const defaults = Object.fromEntries(gdrive.inputs!.map(i => [i.key, i.defaultValue]));
    expect(defaults).toEqual({
      client_id: '${GOOGLE_DRIVE_CLIENT_ID}',
      client_secret: '${GOOGLE_DRIVE_CLIENT_SECRET}',
    });
  });

  it('Stripe is api-key with a required restricted-key → Authorization: Bearer header', () => {
    const stripe = CONNECTOR_CATALOG.find(e => e.id === 'stripe')!;
    expect(stripe.authKind).toBe('api-key');
    expect(stripe.config.auth).toBeUndefined();          // no OAuth block — DCR isn't supported
    expect(stripe.inputs![0].required).toBe(true);
    const cfg = applyQuickInputs(stripe, { restricted_key: 'rk_test_123' });
    expect(cfg.headers).toMatchObject({ Authorization: 'Bearer rk_test_123' });
  });

  it('Exa sends the key as an x-api-key header (not a URL query param)', () => {
    const exa = CONNECTOR_CATALOG.find(e => e.id === 'exa')!;
    expect(exa.authKind).toBe('api-key');
    const cfg = applyQuickInputs(exa, { exa_api_key: 'exa_abc' });
    expect(cfg.headers).toMatchObject({ 'x-api-key': 'exa_abc' });
    expect(cfg.url).toBe('https://mcp.exa.ai/mcp');       // key never leaks into the URL
  });
});

describe('catalogEntryConfigured', () => {
  const gdrive = CONNECTOR_CATALOG.find(e => e.id === 'google-drive')!;
  const fs = CONNECTOR_CATALOG.find(e => e.id === 'filesystem')!;

  it('matches a remote entry by URL, trailing-slash-insensitive', () => {
    expect(catalogEntryConfigured(gdrive, [{ url: 'https://drivemcp.googleapis.com/mcp/v1/' }])).toBe(true);
    expect(catalogEntryConfigured(gdrive, [{ url: 'https://other.example.com/mcp' }])).toBe(false);
    expect(catalogEntryConfigured(gdrive, [])).toBe(false);
  });

  it('matches a local entry by command + package even with different args', () => {
    expect(catalogEntryConfigured(fs, [{
      command: 'npx',
      args: ['-y', '@modelcontextprotocol/server-filesystem', '/home/me/docs'],
    }])).toBe(true);
    expect(catalogEntryConfigured(fs, [{ command: 'npx', args: ['-y', '@other/server'] }])).toBe(false);
  });
});

describe('findConfiguredServer', () => {
  const gdrive = CONNECTOR_CATALOG.find(e => e.id === 'google-drive')!;

  it('returns the backing server object (name may differ from the suggested one)', () => {
    const servers = [
      { name: 'my-gdrive', url: 'https://drivemcp.googleapis.com/mcp/v1', status: 'connected' },
      { name: 'other', url: 'https://x/mcp' },
    ];
    const hit = findConfiguredServer(gdrive, servers);
    expect(hit?.name).toBe('my-gdrive');
    expect(hit?.status).toBe('connected');
  });

  it('returns undefined when nothing matches', () => {
    expect(findConfiguredServer(gdrive, [{ name: 'x', url: 'https://x/mcp' }])).toBeUndefined();
  });
});

describe('draftFromRegistryResult', () => {
  const base: MCPRegistryResult = {
    name: 'io.github.owner/thing',
    description: 'd',
    version: '1.0.0',
    repository_url: 'https://github.com/owner/thing',
    remotes: [],
    packages: [],
  };

  it('prefers a remote endpoint and maps sse/streamable-http transports', () => {
    const sse = draftFromRegistryResult({ ...base, remotes: [{ type: 'sse', url: 'https://x/sse' }] });
    expect(sse.config).toMatchObject({ transport: 'sse', url: 'https://x/sse' });
    const http = draftFromRegistryResult({ ...base, remotes: [{ type: 'streamable-http', url: 'https://x/mcp' }] });
    expect(http.config).toMatchObject({ transport: 'streamable_http', url: 'https://x/mcp' });
  });

  it('maps packages per ecosystem (npm→npx, pypi→uvx, oci→docker)', () => {
    const npm = draftFromRegistryResult({ ...base, packages: [{ registry_type: 'npm', identifier: '@scope/pkg' }] });
    expect(npm.config).toMatchObject({ transport: 'stdio', command: 'npx', args: ['-y', '@scope/pkg'] });
    const pypi = draftFromRegistryResult({ ...base, packages: [{ registry_type: 'pypi', identifier: 'pkg' }] });
    expect(pypi.config).toMatchObject({ command: 'uvx', args: ['pkg'] });
    const oci = draftFromRegistryResult({ ...base, packages: [{ registry_type: 'oci', identifier: 'ghcr.io/o/i:1' }] });
    expect(oci.config).toMatchObject({ command: 'docker', args: ['run', '-i', '--rm', 'ghcr.io/o/i:1'] });
  });

  it('carries review guidance with the repository link', () => {
    const d = draftFromRegistryResult({ ...base, remotes: [{ type: 'streamable-http', url: 'https://x/mcp' }] });
    expect(d.guidance?.docsUrl).toBe(base.repository_url);
    expect(d.guidance?.note).toContain('review');
  });
});

describe('registryShortName', () => {
  it('takes the last path segment and sanitizes it', () => {
    expect(registryShortName('io.github.owner/My Thing!')).toBe('my-thing-');
    expect(registryShortName('plain')).toBe('plain');
  });
});
