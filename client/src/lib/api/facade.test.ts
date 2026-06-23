import { describe, it, expect } from 'vitest';
import { api } from './index';

/**
 * Guards the facade assembly: every domain module must contribute its methods
 * to the single `api` object. One representative method per domain module is
 * asserted present, plus the total method count, so a forgotten spread or a
 * dropped method fails loudly.
 */

// One representative method per domain module (./health … ./streaming).
const REPRESENTATIVE_METHODS = [
  'health',                 // health.ts (Health + Version)
  'authStatus',             // auth.ts
  'listProviders',          // providers.ts
  'listMCPServers',         // mcp.ts
  'runAgent',               // agent.ts
  'enqueueBackgroundChat',  // relay.ts
  'listToolOutputs',        // toolOutput.ts
  'translate',              // translation.ts
  'listPromptProfiles',     // prompts.ts
  'listPromptTemplates',    // promptTemplates.ts
  'listAgentProfiles',      // profiles.ts
  'listAlloyWorkflows',     // alloy.ts
  'getConfig',              // config.ts
  'listMemoryChannels',     // memory.ts
  'listJobs',               // jobs.ts
  'listConversations',      // history.ts
  'streamChat',             // streaming.ts
  'listLogs',               // logs.ts
  'listWorkspaces',         // workspaces.ts
] as const;

describe('api facade', () => {
  it.each(REPRESENTATIVE_METHODS)('exposes %s as a function', (method) => {
    expect(typeof (api as Record<string, unknown>)[method]).toBe('function');
  });

  it('exposes exactly 150 methods (no drops or duplicates from the split)', () => {
    expect(Object.keys(api)).toHaveLength(150);
  });

  it('exposes only functions', () => {
    for (const key of Object.keys(api)) {
      expect(typeof (api as Record<string, unknown>)[key]).toBe('function');
    }
  });
});
