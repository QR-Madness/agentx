import { describe, it, expect } from 'vitest';
import { connectorsNeedingAuth, connectorAuthMessage } from './connectors';
import type { MCPServer } from './api';

function srv(over: Partial<MCPServer>): MCPServer {
  return {
    name: 'x', status: 'disconnected',
    auth: { type: 'oauth' }, auth_state: { authorized: false, pending: false, error: null },
    allowed_agent_ids: null,
    ...over,
  } as MCPServer;
}

describe('connectorsNeedingAuth', () => {
  it('flags an unauthorized OAuth server allowed for all agents', () => {
    expect(connectorsNeedingAuth([srv({ name: 'sentry' })], 'agent-1').map(s => s.name)).toEqual(['sentry']);
  });

  it('excludes authorized / pending / connected / non-oauth servers', () => {
    const servers = [
      srv({ name: 'authed', auth_state: { authorized: true, pending: false, error: null } }),
      srv({ name: 'pending', auth_state: { authorized: false, pending: true, error: null } }),
      srv({ name: 'connected', status: 'connected' }),
      srv({ name: 'static', auth: null }),
      srv({ name: 'needs' }),
    ];
    expect(connectorsNeedingAuth(servers, 'agent-1').map(s => s.name)).toEqual(['needs']);
  });

  it('respects allowed_agent_ids (includes / empty / null)', () => {
    const included = srv({ name: 'inc', allowed_agent_ids: ['agent-1'] });
    const other = srv({ name: 'oth', allowed_agent_ids: ['agent-2'] });
    const none = srv({ name: 'none', allowed_agent_ids: [] });
    const all = srv({ name: 'all', allowed_agent_ids: null });
    const servers = [included, other, none, all];
    expect(connectorsNeedingAuth(servers, 'agent-1').map(s => s.name).sort()).toEqual(['all', 'inc']);
    // A null active agent only matches "all" servers.
    expect(connectorsNeedingAuth(servers, null).map(s => s.name)).toEqual(['all']);
  });
});

describe('connectorAuthMessage', () => {
  it('names a single connector', () => {
    expect(connectorAuthMessage([srv({ name: 'sentry' })])).toBe('sentry needs authorization to use its tools.');
  });
  it('counts multiple connectors', () => {
    expect(connectorAuthMessage([srv({ name: 'a' }), srv({ name: 'b' })])).toBe('2 connectors need authorization.');
  });
});
