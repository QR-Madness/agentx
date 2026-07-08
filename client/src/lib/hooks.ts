/**
 * React hooks for API data fetching
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import {
  api,
  toApiError,
  HealthResponse,
  ProviderInfo,
  ProvidersHealthResponse,
  MCPServer,
  MCPServerConfigInput,
  MCPTool,
  ApiError,
  MemoryChannel,
  MemoryEntity,
  MemoryFact,
  MemoryFactEntity,
  MemoryFactPatch,
  FactForgetResult,
  FactProvenance,
  MemoryEntityPatch,
  MemoryStats,
  UsageMetrics,
  CheckpointsResponse,
  ConversationStateResponse,
  EntityGraph,
  ConsolidateResult,
  MemoryExport,
  MemoryImportResult,
} from './api';

export interface UseApiOptions {
  /** When false, the call is never fired and loading settles to false. */
  enabled?: boolean;
  /** Fetch automatically on mount / when deps change (default true). */
  immediate?: boolean;
  /** Side-effect for errors (e.g. raise a toast); error is still returned. */
  onError?: (error: ApiError) => void;
  /** When set, re-fetches on this interval (ms) while the component is mounted.
   *  Used by status surfaces (health/providers/MCP/agent) so connection state
   *  reflects reality without a manual refresh. Polling pauses while the tab
   *  is hidden, and resumes (with an immediate refresh) when it becomes visible. */
  pollInterval?: number;
}

export interface UseApiResult<T> {
  data: T | null;
  loading: boolean;
  error: ApiError | null;
  refresh: () => Promise<void>;
}

/**
 * Generic data-fetching hook — the shared `data / loading / error / refresh`
 * machinery that the read hooks below all need. Errors are normalized through
 * `toApiError`, so callers always get a structured {@link ApiError}.
 *
 * `call` is read through a ref so it may close over fresh values each render
 * without changing `refresh`'s identity; pass the values it depends on in
 * `deps` to drive re-fetching.
 */
export function useApi<T>(
  call: () => Promise<T>,
  deps: readonly unknown[] = [],
  opts: UseApiOptions = {}
): UseApiResult<T> {
  const { enabled = true, immediate = true } = opts;
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(enabled && immediate);
  const [error, setError] = useState<ApiError | null>(null);

  const callRef = useRef(call);
  callRef.current = call;
  const onErrorRef = useRef(opts.onError);
  onErrorRef.current = opts.onError;

  const refresh = useCallback(async () => {
    if (!enabled) return;
    setLoading(true);
    setError(null);
    try {
      setData(await callRef.current());
    } catch (err) {
      const apiErr = toApiError(err);
      setError(apiErr);
      onErrorRef.current?.(apiErr);
    } finally {
      setLoading(false);
    }
  // deps intentionally spread to re-bind refresh when inputs change
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [enabled, ...deps]);

  useEffect(() => {
    if (immediate && enabled) refresh();
  }, [refresh, immediate, enabled]);

  // Polling: re-fetch every `pollInterval` ms while the tab is visible. We
  // skip the wakeup tick when document.hidden is true (saves cycles for
  // backgrounded windows) and trigger a one-shot refresh on visibilitychange
  // so the user always sees current state right after switching back.
  const { pollInterval } = opts;
  useEffect(() => {
    if (!enabled || !pollInterval || pollInterval <= 0) return;
    const tick = () => {
      if (typeof document !== 'undefined' && document.hidden) return;
      refresh();
    };
    const id = setInterval(tick, pollInterval);
    const onVisible = () => {
      if (typeof document !== 'undefined' && !document.hidden) refresh();
    };
    if (typeof document !== 'undefined') {
      document.addEventListener('visibilitychange', onVisible);
    }
    return () => {
      clearInterval(id);
      if (typeof document !== 'undefined') {
        document.removeEventListener('visibilitychange', onVisible);
      }
    };
  }, [refresh, enabled, pollInterval]);

  return { data, loading, error, refresh };
}

export function useHealth(includeMemory = true, includeStorage = true, opts?: UseApiOptions) {
  return useApi<HealthResponse>(
    () => api.health(includeMemory, includeStorage),
    [includeMemory, includeStorage],
    opts
  );
}

export function useProviders(opts?: UseApiOptions) {
  const { data, loading, error, refresh } = useApi<{ providers: ProviderInfo[] }>(
    () => api.listProviders(),
    [],
    opts
  );
  return { providers: data?.providers ?? [], loading, error, refresh };
}

/** Live per-provider reachability — pings each configured provider in parallel.
 *  Use this (not useProviders) when you need to know if calls would actually succeed. */
export function useProvidersHealth(opts?: UseApiOptions) {
  const { data, loading, error, refresh } = useApi<ProvidersHealthResponse>(
    () => api.checkProvidersHealth(),
    [],
    opts
  );
  return {
    overall: data?.status,
    providers: data?.providers ?? {},
    loading,
    error,
    refresh,
  };
}

export function useMCPServers(opts?: { pollInterval?: number }) {
  const [servers, setServers] = useState<MCPServer[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<ApiError | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const result = await api.listMCPServers();
      setServers(result.servers || []);
    } catch (err) {
      setError(err as ApiError);
    } finally {
      setLoading(false);
    }
  }, []);

  const connectServer = useCallback(async (name: string) => {
    // Returns the connect result so callers can drive the OAuth round-trip:
    // `auth_required` carries the consent URL to open; the server transitions
    // to connected in the background (visible via refresh/poll).
    try {
      const result = await api.connectMCPServer(name);
      await refresh();
      return result;
    } catch (err) {
      setError(err as ApiError);
      return { status: 'error', server: name };
    }
  }, [refresh]);

  const connectAll = useCallback(async () => {
    try {
      await api.connectAllMCPServers();
      await refresh();
    } catch (err) {
      setError(err as ApiError);
    }
  }, [refresh]);

  const disconnectServer = useCallback(async (name: string) => {
    try {
      await api.disconnectMCPServer(name);
      await refresh();
    } catch (err) {
      setError(err as ApiError);
    }
  }, [refresh]);

  const createServer = useCallback(async (name: string, config: MCPServerConfigInput) => {
    const result = await api.createMCPServer(name, config);
    await refresh();
    return result;
  }, [refresh]);

  const updateServer = useCallback(async (name: string, config: MCPServerConfigInput, rename?: string) => {
    const result = await api.updateMCPServer(name, config, rename);
    await refresh();
    return result;
  }, [refresh]);

  const deleteServer = useCallback(async (name: string) => {
    const result = await api.deleteMCPServer(name);
    await refresh();
    return result;
  }, [refresh]);

  const validateServer = useCallback(async (name: string, config: MCPServerConfigInput) => {
    return api.validateMCPServer(name, config);
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  // Optional polling so MCP connection status (connected / error / disconnected)
  // updates without a manual refresh. Mirrors the pollInterval behavior in
  // useApi: skip the tick when the tab is hidden, refresh on visibility return.
  const pollInterval = opts?.pollInterval;
  useEffect(() => {
    if (!pollInterval || pollInterval <= 0) return;
    const tick = () => {
      if (typeof document !== 'undefined' && document.hidden) return;
      refresh();
    };
    const id = setInterval(tick, pollInterval);
    const onVisible = () => {
      if (typeof document !== 'undefined' && !document.hidden) refresh();
    };
    if (typeof document !== 'undefined') {
      document.addEventListener('visibilitychange', onVisible);
    }
    return () => {
      clearInterval(id);
      if (typeof document !== 'undefined') {
        document.removeEventListener('visibilitychange', onVisible);
      }
    };
  }, [refresh, pollInterval]);

  return {
    servers, loading, error, refresh,
    connectServer, connectAll, disconnectServer,
    createServer, updateServer, deleteServer, validateServer,
  };
}

export function useMCPTools() {
  const { data, loading, error, refresh } = useApi<{ tools: MCPTool[] }>(
    () => api.listMCPTools(),
    []
  );
  return { tools: data?.tools ?? [], loading, error, refresh };
}

export function useAgentStatus(opts?: UseApiOptions) {
  const { data, loading, error, refresh } = useApi<{ status: string; active_sessions: number }>(
    () => api.getAgentStatus(),
    [],
    opts
  );
  return { status: data, loading, error, refresh };
}

// === Memory Hooks ===

export function useMemoryChannels() {
  const { data, loading, error, refresh } = useApi<{ channels: MemoryChannel[] }>(
    () => api.listMemoryChannels(),
    []
  );
  return { channels: data?.channels ?? [], loading, error, refresh };
}

export function useMemoryEntities(
  channel: string,
  page: number,
  search?: string,
  type?: string
) {
  const { data, loading, error, refresh } = useApi(
    () => api.listMemoryEntities({
      channel,
      page,
      search: search || undefined,
      type: type || undefined,
    }),
    [channel, page, search, type]
  );
  return {
    entities: data?.entities ?? [],
    total: data?.total ?? 0,
    hasNext: data?.has_next ?? false,
    loading,
    error,
    refresh,
  };
}

export function useEntityGraph(entityId: string | null) {
  const { data, loading, error, refresh } = useApi<EntityGraph>(
    () => api.getEntityGraph(entityId as string),
    [entityId],
    { enabled: !!entityId }
  );
  // Mirror the prior behavior of clearing the graph when no entity is selected.
  return { graph: entityId ? data : null, loading, error, refresh };
}

export function useMemoryFacts(
  channel: string,
  page: number,
  minConfidence?: number,
  search?: string
) {
  const { data, loading, error, refresh } = useApi(
    () => api.listMemoryFacts({
      channel,
      page,
      min_confidence: minConfidence,
      search: search || undefined,
    }),
    [channel, page, minConfidence, search]
  );
  return {
    facts: data?.facts ?? [],
    total: data?.total ?? 0,
    hasNext: data?.has_next ?? false,
    loading,
    error,
    refresh,
  };
}

export function useMemoryStrategies(channel: string, page: number) {
  const { data, loading, error, refresh } = useApi(
    () => api.listMemoryStrategies({ channel, page }),
    [channel, page]
  );
  return {
    strategies: data?.strategies ?? [],
    total: data?.total ?? 0,
    hasNext: data?.has_next ?? false,
    loading,
    error,
    refresh,
  };
}

export function useMemoryProcedures(channel: string, page: number) {
  const { data, loading, error, refresh } = useApi(
    () => api.listMemoryProcedures({ channel, page }),
    [channel, page]
  );
  return {
    procedures: data?.procedures ?? [],
    total: data?.total ?? 0,
    hasNext: data?.has_next ?? false,
    loading,
    error,
    refresh,
  };
}

export function useMemoryStats(opts?: UseApiOptions) {
  const { data, loading, error, refresh } = useApi<MemoryStats>(
    () => api.getMemoryStats(),
    [],
    opts
  );
  return { stats: data, loading, error, refresh };
}

export function useUsageMetrics(days = 14) {
  const { data, loading, error, refresh } = useApi<UsageMetrics>(
    () => api.getUsageMetrics(days),
    [days]
  );
  return { usage: data, loading, error, refresh };
}

export function useCheckpoints(conversationId: string | null | undefined) {
  const { data, loading, error, refresh } = useApi<CheckpointsResponse>(
    () => api.getCheckpoints(conversationId as string),
    [conversationId],
    { enabled: !!conversationId },
  );
  return {
    checkpoints: data?.checkpoints ?? [],
    count: data?.count ?? 0,
    loading,
    error,
    refresh,
  };
}

export function useConversationState(conversationId: string | null | undefined) {
  const { data, loading, error, refresh } = useApi<ConversationStateResponse>(
    () => api.getConversationState(conversationId as string),
    [conversationId],
    { enabled: !!conversationId },
  );
  const state = data?.state;
  const counts = {
    goals: state?.goals.length ?? 0,
    decisions: state?.decisions.length ?? 0,
    open_threads: state?.open_threads.length ?? 0,
    artifacts: state?.artifacts.length ?? 0,
    narrative: state?.narrative.length ?? 0,
  };
  const total =
    counts.goals + counts.decisions + counts.open_threads + counts.artifacts + counts.narrative;
  return { state, counts, total, loading, error, refresh };
}

// === Memory Mutation Hooks ===
//
// Each hook returns { mutate, loading, error }. Callers are responsible for
// triggering refresh() on the relevant list hook (useMemoryFacts /
// useMemoryEntities / useMemoryStats) after a successful mutation —
// matching the existing manual-refresh pattern used elsewhere in this file.

export function useUpdateMemoryFact() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<ApiError | null>(null);

  const mutate = useCallback(async (factId: string, patch: MemoryFactPatch): Promise<MemoryFact | null> => {
    setLoading(true);
    setError(null);
    try {
      const result = await api.updateMemoryFact(factId, patch);
      return result.fact;
    } catch (err) {
      setError(err as ApiError);
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  return { mutate, loading, error };
}

export function useDeleteMemoryFact() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<ApiError | null>(null);

  const mutate = useCallback(async (factId: string): Promise<boolean> => {
    setLoading(true);
    setError(null);
    try {
      const result = await api.deleteMemoryFact(factId);
      return result.deleted;
    } catch (err) {
      setError(err as ApiError);
      return false;
    } finally {
      setLoading(false);
    }
  }, []);

  return { mutate, loading, error };
}

export function useRememberMemoryFact() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<ApiError | null>(null);

  const mutate = useCallback(async (factId: string, to?: number): Promise<MemoryFact | null> => {
    setLoading(true);
    setError(null);
    try {
      const result = await api.rememberMemoryFact(factId, to);
      return result.fact;
    } catch (err) {
      setError(err as ApiError);
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  return { mutate, loading, error };
}

export function useForgetMemoryFact() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<ApiError | null>(null);

  const mutate = useCallback(async (factId: string, hard = false): Promise<FactForgetResult | null> => {
    setLoading(true);
    setError(null);
    try {
      return await api.forgetMemoryFact(factId, hard);
    } catch (err) {
      setError(err as ApiError);
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  return { mutate, loading, error };
}

export function useLinkFactEntity() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<ApiError | null>(null);

  const mutate = useCallback(
    async (factId: string, entityId: string): Promise<MemoryFactEntity[] | null> => {
      setLoading(true);
      setError(null);
      try {
        const result = await api.linkFactEntity(factId, entityId);
        return result.entities;
      } catch (err) {
        setError(err as ApiError);
        return null;
      } finally {
        setLoading(false);
      }
    },
    [],
  );

  return { mutate, loading, error };
}

export function useUnlinkFactEntity() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<ApiError | null>(null);

  const mutate = useCallback(
    async (factId: string, entityId: string): Promise<MemoryFactEntity[] | null> => {
      setLoading(true);
      setError(null);
      try {
        const result = await api.unlinkFactEntity(factId, entityId);
        return result.entities;
      } catch (err) {
        setError(err as ApiError);
        return null;
      } finally {
        setLoading(false);
      }
    },
    [],
  );

  return { mutate, loading, error };
}

export function useFactProvenance() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<ApiError | null>(null);

  const fetch = useCallback(async (factId: string): Promise<FactProvenance | null> => {
    setLoading(true);
    setError(null);
    try {
      return await api.getFactProvenance(factId);
    } catch (err) {
      setError(err as ApiError);
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  return { fetch, loading, error };
}

export function useUpdateMemoryEntity() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<ApiError | null>(null);

  const mutate = useCallback(async (entityId: string, patch: MemoryEntityPatch): Promise<MemoryEntity | null> => {
    setLoading(true);
    setError(null);
    try {
      const result = await api.updateMemoryEntity(entityId, patch);
      return result.entity;
    } catch (err) {
      setError(err as ApiError);
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  return { mutate, loading, error };
}

export function useDeleteMemoryEntity() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<ApiError | null>(null);

  const mutate = useCallback(async (entityId: string): Promise<boolean> => {
    setLoading(true);
    setError(null);
    try {
      const result = await api.deleteMemoryEntity(entityId);
      return result.deleted;
    } catch (err) {
      setError(err as ApiError);
      return false;
    } finally {
      setLoading(false);
    }
  }, []);

  return { mutate, loading, error };
}

// === Job Hooks ===

export function useJobs() {
  const { data, loading, error, refresh } = useApi(() => api.listJobs(), []);
  return { jobs: data?.jobs ?? [], worker: data?.worker ?? null, loading, error, refresh };
}

export function useJob(name: string) {
  const { data, loading, error, refresh } = useApi(
    () => api.getJob(name),
    [name],
    { enabled: !!name }
  );
  return { job: data?.job ?? null, history: data?.history ?? [], loading, error, refresh };
}

export function useConsolidate() {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<ConsolidateResult | null>(null);
  const [error, setError] = useState<ApiError | null>(null);

  const consolidate = useCallback(async (jobs?: string[]) => {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await api.consolidateNow(jobs);
      setResult(res);
      return res;
    } catch (err) {
      setError(err as ApiError);
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);

  const reset = useCallback(() => {
    setResult(null);
    setError(null);
  }, []);

  return { consolidate, loading, result, error, reset };
}

export function useExportMemory() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<ApiError | null>(null);

  const mutate = useCallback(
    async (params: { channel?: string } = {}): Promise<MemoryExport | null> => {
      setLoading(true);
      setError(null);
      try {
        const res = await api.exportMemory(params);
        return res.export;
      } catch (err) {
        setError(err as ApiError);
        return null;
      } finally {
        setLoading(false);
      }
    },
    [],
  );

  return { mutate, loading, error };
}

export function useImportMemory() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<ApiError | null>(null);

  const mutate = useCallback(
    async (
      data: MemoryExport,
      mode: 'merge' | 'replace' = 'merge',
      channel?: string,
    ): Promise<MemoryImportResult | null> => {
      setLoading(true);
      setError(null);
      try {
        const res = await api.importMemory(data, mode, channel);
        return res.imported;
      } catch (err) {
        setError(err as ApiError);
        return null;
      } finally {
        setLoading(false);
      }
    },
    [],
  );

  return { mutate, loading, error };
}

/** Save status exposed by {@link useSettingsAutosave} (drives Saving…/Saved ✓). */
export type SettingsSaveStatus = 'idle' | 'saving' | 'saved' | 'error';

export interface UseSettingsAutosaveResult<T> {
  /** Draft = last server state + local edits (render forms from this). */
  settings: T | null;
  loading: boolean;
  /** Load or save error, whichever is most recent. */
  error: ApiError | null;
  status: SettingsSaveStatus;
  /** Convenience: status === 'saving'. */
  saving: boolean;
  /** Stage edits; a debounced baseline-diff save follows automatically. */
  update: (patch: Partial<T>) => void;
  /** Save any pending edits immediately. Resolves false on failure. */
  flush: () => Promise<boolean>;
  refresh: () => Promise<void>;
}

/**
 * useSettingsAutosave — the ONE save pattern for preference sections
 * (settings overhaul D5: autosave everywhere; secrets keep explicit Save).
 *
 * Loads via {@link useApi}, keeps a local draft, and after `debounceMs` of
 * inactivity persists only the keys that differ from the last server state
 * (baseline-diff — no no-op writes). Pending edits flush on unmount so a
 * quick navigation away can't drop a change.
 */
export function useSettingsAutosave<T extends Record<string, unknown>>(opts: {
  load: () => Promise<T>;
  save: (updates: Partial<T>) => Promise<unknown>;
  enabled?: boolean;
  debounceMs?: number;
  /** Side-effect for load/save errors (e.g. `notifyError`). */
  onError?: (error: ApiError) => void;
}): UseSettingsAutosaveResult<T> {
  const { load, save, enabled = true, debounceMs = 800, onError } = opts;
  const { data, loading, error: loadError, refresh } = useApi<T>(load, [], { enabled, onError });

  const [draft, setDraft] = useState<T | null>(null);
  const [status, setStatus] = useState<SettingsSaveStatus>('idle');
  const [saveError, setSaveError] = useState<ApiError | null>(null);

  const baselineRef = useRef<T | null>(null);
  const draftRef = useRef<T | null>(null);
  draftRef.current = draft;
  const pendingRef = useRef<Partial<T>>({});
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const savedTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const saveRef = useRef(save);
  saveRef.current = save;
  const onErrorRef = useRef(onError);
  onErrorRef.current = onError;

  // Adopt fresh server state as the baseline; re-apply edits staged since.
  useEffect(() => {
    if (data) {
      baselineRef.current = data;
      setDraft({ ...data, ...pendingRef.current });
    }
  }, [data]);

  const flush = useCallback(async (): Promise<boolean> => {
    if (timerRef.current) {
      clearTimeout(timerRef.current);
      timerRef.current = null;
    }
    const baseline = baselineRef.current;
    const current = draftRef.current;
    if (!baseline || !current) return true;

    const changed: Partial<T> = {};
    for (const key of Object.keys(current) as (keyof T)[]) {
      if (!Object.is(current[key], baseline[key])) changed[key] = current[key];
    }
    if (Object.keys(changed).length === 0) {
      pendingRef.current = {};
      return true;
    }

    setStatus('saving');
    try {
      await saveRef.current(changed);
      baselineRef.current = { ...baseline, ...changed };
      pendingRef.current = {};
      setSaveError(null);
      setStatus('saved');
      if (savedTimerRef.current) clearTimeout(savedTimerRef.current);
      savedTimerRef.current = setTimeout(
        () => setStatus(s => (s === 'saved' ? 'idle' : s)),
        2000
      );
      return true;
    } catch (err) {
      const apiErr = toApiError(err);
      setSaveError(apiErr);
      setStatus('error');
      onErrorRef.current?.(apiErr);
      return false;
    }
  }, []);

  const update = useCallback((patch: Partial<T>) => {
    pendingRef.current = { ...pendingRef.current, ...patch };
    setDraft(prev => (prev ? { ...prev, ...patch } : prev));
    if (timerRef.current) clearTimeout(timerRef.current);
    timerRef.current = setTimeout(() => void flush(), debounceMs);
  }, [flush, debounceMs]);

  // Flush pending edits on unmount (fire-and-forget) so navigating away
  // mid-debounce never drops a change.
  useEffect(() => () => {
    if (savedTimerRef.current) clearTimeout(savedTimerRef.current);
    if (timerRef.current) {
      clearTimeout(timerRef.current);
      void flush();
    }
  }, [flush]);

  return {
    settings: draft,
    loading,
    error: saveError ?? loadError,
    status,
    saving: status === 'saving',
    update,
    flush,
    refresh,
  };
}

/**
 * Hook for streaming consolidation progress via SSE.
 * Provides live progress updates during consolidation runs.
 */
export function useConsolidationStream() {
  const [isStreaming, setIsStreaming] = useState(false);
  const [currentJob, setCurrentJob] = useState<string | null>(null);
  const [currentJobIndex, setCurrentJobIndex] = useState(0);
  const [totalJobs, setTotalJobs] = useState(0);
  const [stage, setStage] = useState<string | null>(null);
  const [progress, setProgress] = useState<Record<string, unknown> | null>(null);
  const [result, setResult] = useState<Record<string, unknown> | null>(null);
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<(() => void) | null>(null);

  const startStream = useCallback((options?: { jobs?: string[]; trigger?: boolean }) => {
    // Abort any existing stream
    abortRef.current?.();

    setIsStreaming(true);
    setCurrentJob(null);
    setCurrentJobIndex(0);
    setTotalJobs(0);
    setStage(null);
    setProgress(null);
    setResult(null);
    setError(null);

    const { abort } = api.streamConsolidate(
      { trigger: options?.trigger ?? true, jobs: options?.jobs },
      {
        onStart: (data) => {
          setTotalJobs(data.total_jobs);
        },
        onJobStart: (data) => {
          setCurrentJob(data.job);
          setCurrentJobIndex(data.index);
          setStage('starting');
        },
        onProgress: (data) => {
          setCurrentJob(data.job);
          setStage(data.stage);
          setProgress(data as unknown as Record<string, unknown>);
        },
        onJobDone: (data) => {
          setStage('done');
          setProgress(data as unknown as Record<string, unknown>);
        },
        onDone: (data) => {
          setIsStreaming(false);
          setResult(data as unknown as Record<string, unknown>);
          setStage(null);
          setCurrentJob(null);
        },
        onIdle: () => {
          setIsStreaming(false);
        },
        onError: (err) => {
          setIsStreaming(false);
          setError(err);
        },
      }
    );

    abortRef.current = abort;
    return { abort };
  }, []);

  const stop = useCallback(() => {
    abortRef.current?.();
    abortRef.current = null;
    setIsStreaming(false);
  }, []);

  return {
    startStream,
    stop,
    isStreaming,
    currentJob,
    currentJobIndex,
    totalJobs,
    stage,
    progress,
    result,
    error,
  };
}

/**
 * Hook that provides consolidation activity status.
 * Checks jobs endpoint on mount, auto-subscribes to SSE if consolidation is active.
 */
export function useConsolidationStatus() {
  const stream = useConsolidationStream();
  const [isActive, setIsActive] = useState(false);
  const [checkedOnMount, setCheckedOnMount] = useState(false);

  // Check for active consolidation on mount
  useEffect(() => {
    let cancelled = false;

    async function checkActive() {
      try {
        const jobsData = await api.listJobs();
        if (cancelled) return;

        if (jobsData.consolidation_active) {
          setIsActive(true);
          // Auto-subscribe to watch progress (GET = watch-only)
          stream.startStream({ trigger: false });
        }
      } catch {
        // Ignore — server may not be available
      } finally {
        if (!cancelled) setCheckedOnMount(true);
      }
    }

    checkActive();
    return () => { cancelled = true; };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Sync isActive with stream state
  useEffect(() => {
    setIsActive(stream.isStreaming);
  }, [stream.isStreaming]);

  const trigger = useCallback((jobs?: string[]) => {
    setIsActive(true);
    return stream.startStream({ trigger: true, jobs });
  }, [stream]);

  return {
    isActive,
    isStreaming: stream.isStreaming,
    currentJob: stream.currentJob,
    currentJobIndex: stream.currentJobIndex,
    totalJobs: stream.totalJobs,
    stage: stream.stage,
    progress: stream.progress,
    result: stream.result,
    error: stream.error,
    trigger,
    stop: stream.stop,
    checkedOnMount,
  };
}


/**
 * Returns true when the viewport matches the mobile breakpoint (max-width: 600px).
 * Mirrors the CSS @media (max-width: 600px) rule used across the layout.
 */
export function useIsMobile(): boolean {
  const [isMobile, setIsMobile] = useState(
    () => typeof window !== "undefined" && window.matchMedia("(max-width: 600px)").matches,
  );
  useEffect(() => {
    const mq = window.matchMedia("(max-width: 600px)");
    const handler = (e: MediaQueryListEvent) => setIsMobile(e.matches);
    mq.addEventListener("change", handler);
    return () => mq.removeEventListener("change", handler);
  }, []);
  return isMobile;
}
