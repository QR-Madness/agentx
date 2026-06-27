import { describe, it, expect } from 'vitest';
import { api } from './index';

/**
 * Guards the facade assembly: every domain module must contribute its methods
 * to the single `api` object. One representative method per domain module is
 * asserted present, plus the full method set is snapshotted so a forgotten
 * spread or a dropped/duplicated method fails loudly.
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

  it('exposes the expected method set (snapshot — review new keys on change)', () => {
    // Snapshot the sorted key list so adding a method is a visible diff in the PR,
    // not a magic number someone has to remember to bump. To intentionally add a
    // method, update both the spread in ./index.ts AND this snapshot.
    expect(Object.keys(api).sort()).toMatchInlineSnapshot(`
      [
        "acknowledgePromptLayer",
        "ambassadorPersonaDefaults",
        "askAmbassador",
        "attachChatRun",
        "authSession",
        "authSetup",
        "authStatus",
        "briefTurn",
        "cancelChatRun",
        "cancelPlan",
        "changePassword",
        "chat",
        "checkProvidersHealth",
        "clearAmbassadorThread",
        "clearCheckpoints",
        "clearStuckJobs",
        "composePrompt",
        "connectAllMCPServers",
        "connectMCPServer",
        "consolidateNow",
        "createAgentProfile",
        "createAlloyWorkflow",
        "createAmbassadorThread",
        "createMCPServer",
        "createPromptLayer",
        "createPromptTemplate",
        "createWorkspace",
        "deleteAgentProfile",
        "deleteAlloyWorkflow",
        "deleteConversation",
        "deleteDocument",
        "deleteMCPServer",
        "deleteMemoryEntity",
        "deleteMemoryFact",
        "deletePromptLayer",
        "deletePromptTemplate",
        "deleteToolOutput",
        "deleteWorkspace",
        "detectLanguage",
        "disconnectMCPServer",
        "dismissBackgroundChat",
        "dispatchAmbassador",
        "downloadLogArchive",
        "draftRelay",
        "enhancePrompt",
        "enqueueBackgroundChat",
        "exportMemory",
        "fetchAmbassadorBriefings",
        "fetchAmbassadorThread",
        "fetchMediaBlob",
        "forgetMemoryFact",
        "generateAvatar",
        "getAgentProfile",
        "getAgentStatus",
        "getAlloyWorkflow",
        "getBackgroundChat",
        "getCheckpoints",
        "getConfig",
        "getConsolidationSettings",
        "getContextLimits",
        "getConversationMessages",
        "getEntityGraph",
        "getFactProvenance",
        "getGlobalPrompt",
        "getJob",
        "getLogArchiveStatus",
        "getLogCategories",
        "getMCPToolsPrompt",
        "getMemoryStats",
        "getPlanStatus",
        "getPromptProfile",
        "getPromptTemplate",
        "getRecallSettings",
        "getToolOutput",
        "getUsageMetrics",
        "getUserHistory",
        "getWorkspace",
        "getWorkspaceContainer",
        "health",
        "importMemory",
        "linkFactEntity",
        "listAgentProfiles",
        "listAlloyWorkflows",
        "listAmbassadorThreads",
        "listBackgroundChats",
        "listChatRuns",
        "listConversations",
        "listDocuments",
        "listJobs",
        "listLogArchive",
        "listLogs",
        "listMCPResources",
        "listMCPServers",
        "listMCPTools",
        "listMemoryChannels",
        "listMemoryEntities",
        "listMemoryFacts",
        "listMemoryProcedures",
        "listMemoryStrategies",
        "listModels",
        "listPromptLayers",
        "listPromptProfiles",
        "listPromptSections",
        "listPromptTemplateTags",
        "listPromptTemplates",
        "listProviders",
        "listToolOutputs",
        "listWorkspaces",
        "logArchiveUrl",
        "login",
        "logout",
        "relayAmbassador",
        "rememberMemoryFact",
        "renameAmbassadorThread",
        "renameWorkspace",
        "reorderPromptLayers",
        "resetMemory",
        "resetPromptLayer",
        "resetPromptTemplate",
        "resumePlan",
        "runAgent",
        "runJob",
        "searchHealth",
        "setDefaultAgentProfile",
        "setDefaultAmbassador",
        "setWorkspaceShell",
        "setWorkspaceShellBackend",
        "speak",
        "steerChatRun",
        "streamAmbassador",
        "streamChat",
        "streamConsolidate",
        "streamLogs",
        "toggleJob",
        "transcribe",
        "translate",
        "unlinkFactEntity",
        "updateAgentProfile",
        "updateAlloyWorkflow",
        "updateConfig",
        "updateConsolidationSettings",
        "updateContextLimits",
        "updateGlobalPrompt",
        "updateMCPServer",
        "updateMemoryEntity",
        "updateMemoryFact",
        "updatePromptLayer",
        "updatePromptTemplate",
        "updateRecallSettings",
        "uploadChatImage",
        "uploadDocument",
        "validateMCPServer",
        "version",
        "voiceCommand",
        "workspaceContainerAction",
      ]
    `);
  });

  it('exposes only functions', () => {
    for (const key of Object.keys(api)) {
      expect(typeof (api as Record<string, unknown>)[key]).toBe('function');
    }
  });
});
