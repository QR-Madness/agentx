/**
 * API client for AgentX backend — public facade.
 *
 * All calls are routed through the active server configuration. The client is
 * split into per-domain modules (see `./health`, `./mcp`, `./memory`, …) that
 * share the request layer in `./core`; this barrel assembles them into the
 * single `api` object and re-exports the public types + error/version helpers,
 * so consumers keep importing everything from `'.../lib/api'` unchanged.
 */

import { healthApi } from './health';
import { authApi } from './auth';
import { providersApi } from './providers';
import { mcpApi } from './mcp';
import { agentApi } from './agent';
import { relayApi } from './relay';
import { toolOutputApi } from './toolOutput';
import { translationApi } from './translation';
import { promptsApi } from './prompts';
import { promptTemplatesApi } from './promptTemplates';
import { profilesApi } from './profiles';
import { alloyApi } from './alloy';
import { configApi } from './config';
import { memoryApi } from './memory';
import { metricsApi } from './metrics';
import { jobsApi } from './jobs';
import { historyApi } from './history';
import { streamingApi } from './streaming';
import { ambassadorApi } from './ambassador';

/**
 * The API client singleton. Flat method surface (`api.health()`, `api.chat()`,
 * `api.recall()`, `api.streamChat()`, …) assembled from the domain modules.
 * Method names are unique across domains, so the spread never collides.
 */
export const api = {
  ...healthApi,
  ...authApi,
  ...providersApi,
  ...mcpApi,
  ...agentApi,
  ...relayApi,
  ...toolOutputApi,
  ...translationApi,
  ...promptsApi,
  ...promptTemplatesApi,
  ...profilesApi,
  ...alloyApi,
  ...configApi,
  ...memoryApi,
  ...metricsApi,
  ...jobsApi,
  ...historyApi,
  ...streamingApi,
  ...ambassadorApi,
};

// Public surface re-exports (preserve the `'.../lib/api'` specifier).
export * from './types';
export type {
  AmbassadorBriefing,
  AmbassadorQA,
  AmbassadorStatus,
  AmbassadorStreamCallbacks,
  AmbassadorTurnArtifacts,
  AmbassadorToolArtifact,
  AmbassadorSource,
  AmbassadorExhibitArtifact,
  AskAmbassadorRequest,
  DraftRelayRequest,
  BriefTurnRequest,
} from './ambassador';
export * from './errors';
export * from './version';
export { setAuthRequired } from './core';
