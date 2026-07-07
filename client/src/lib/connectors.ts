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

/**
 * OAuth connectors that need the user to sign in before their tools work for
 * the active agent: OAuth auth, not already authorized/pending/connected, and
 * allowed for this agent.
 */
export function connectorsNeedingAuth(
  servers: MCPServer[],
  activeAgentId: string | null,
): MCPServer[] {
  return servers.filter(s =>
    s.auth?.type === 'oauth'
    && !s.auth_state?.authorized
    && !s.auth_state?.pending
    && s.status !== 'connected'
    && allowedForAgent(s, activeAgentId)
  );
}

/** One-line nudge copy for a set of connectors needing auth. */
export function connectorAuthMessage(servers: MCPServer[]): string {
  if (servers.length === 1) {
    return `${servers[0].name} needs authorization to use its tools.`;
  }
  return `${servers.length} connectors need authorization.`;
}
