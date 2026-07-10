import { describe, it, expect } from 'vitest';
import type { MCPRegistryResult } from './api';
import {
  CONNECTOR_CATALOG, catalogEntryConfigured, draftFromCatalogEntry,
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
