/**
 * Connector (OAuth MCP server) helpers — shared logic for surfacing servers
 * that still need an interactive sign-in. Kept pure so the new-conversation
 * nudge (ConnectorAuthNudge) is trivially testable.
 */
import type { MCPServer } from './api';

/** Is this server allowed for the given agent? Mirrors the Toolkit AccessView:
 *  `allowed_agent_ids == null` → all agents; an explicit list must include the
 *  agent's `agentId`; an **empty** list means no agents. */
function allowedForAgent(server: MCPServer, activeAgentId: string | null): boolean {
  const allowed = server.allowed_agent_ids;
  if (allowed == null) return true;
  return activeAgentId != null && allowed.includes(activeAgentId);
}

/** A stored session that cannot come back headlessly: access token expired and
 *  no refresh_token — the next connect must go through the browser again. */
export function sessionExpired(server: MCPServer): boolean {
  const a = server.auth_state;
  return !!a?.authorized && a.expired === true && !a.refreshable;
}

/** Agent-agnostic auth need: an OAuth server, not pending/connected, that is
 *  either never authorized or holding an expired session that can't refresh
 *  itself. (The per-agent nudge adds the whitelist filter on top.) */
export function needsAuth(server: MCPServer): boolean {
  return server.auth?.type === 'oauth'
    && (!server.auth_state?.authorized || sessionExpired(server))
    && !server.auth_state?.pending
    && server.status !== 'connected';
}

/**
 * OAuth connectors that need the user to sign in before their tools work for
 * the active agent: `needsAuth` plus allowed for this agent.
 */
export function connectorsNeedingAuth(
  servers: MCPServer[],
  activeAgentId: string | null,
): MCPServer[] {
  return servers.filter(s => needsAuth(s) && allowedForAgent(s, activeAgentId));
}

/** One-line nudge copy for a set of connectors needing auth. */
export function connectorAuthMessage(servers: MCPServer[]): string {
  if (servers.length === 1) {
    const s = servers[0];
    return sessionExpired(s)
      ? `${s.name}'s session expired — sign in again to use its tools.`
      : `${s.name} needs authorization to use its tools.`;
  }
  return `${servers.length} connectors need authorization.`;
}
