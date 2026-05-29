/**
 * React hooks for API data fetching
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import { useAuth } from '../contexts/AuthContext';
import {
  api,
  toApiError,
  HealthResponse,
  ProviderInfo,
  MCPServer,
  MCPServerConfigInput,
  MCPTool,
  ApiError,
  MemoryChannel,
  MemoryEntity,
  MemoryFact,
  MemoryFactPatch,
  MemoryEntityPatch,
  MemoryStats,
  UsageMetrics,
  CheckpointsResponse,
  EntityGraph,
  ConsolidateResult,
  ConsolidationSettings,
  RecallSettings,
} from './api';

export interface UseApiOptions {
  /** When false, the call is never fired and loading settles to false. */
  enabled?: boolean;
  /** Fetch automatically on mount / when deps change (default true). */
  immediate?: boolean;
  /** Side-effect for errors (e.g. raise a toast); error is still returned. */
  onError?: (error: ApiError) => void;
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

  return { data, loading, error, refresh };
}

export function useHealth(includeMemory = true, includeStorage = true) {
  return useApi<HealthResponse>(
    () => api.health(includeMemory, includeStorage),
    [includeMemory, includeStorage]
  );
}

export function useProviders() {
  const { data, loading, error, refresh } = useApi<{ providers: ProviderInfo[] }>(
    () => api.listProviders(),
    []
  );
  return { providers: data?.providers ?? [], loading, error, refresh };
}

export function useMCPServers() {
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
    try {
      await api.connectMCPServer(name);
      await refresh();
    } catch (err) {
      setError(err as ApiError);
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

export function useAgentStatus() {
  const { data, loading, error, refresh } = useApi<{ status: string; active_sessions: number }>(
    () => api.getAgentStatus(),
    []
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

export function useMemoryStats() {
  const { data, loading, error, refresh } = useApi<MemoryStats>(
    () => api.getMemoryStats(),
    []
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

export function useConsolidationSettings() {
  const { isAuthenticated, authRequired, isLoading: authLoading } = useAuth();
  const [settings, setSettings] = useState<ConsolidationSettings | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<ApiError | null>(null);

  const fetchSettings = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await api.getConsolidationSettings();
      setSettings(res);
    } catch (err) {
      setError(err as ApiError);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (authLoading) return;
    if (authRequired && !isAuthenticated) return;
    fetchSettings();
  }, [fetchSettings, authLoading, authRequired, isAuthenticated]);

  const updateSettings = useCallback(async (updates: Partial<ConsolidationSettings>) => {
    setSaving(true);
    setError(null);
    try {
      await api.updateConsolidationSettings(updates);
      // Refresh settings after save
      await fetchSettings();
      return true;
    } catch (err) {
      setError(err as ApiError);
      return false;
    } finally {
      setSaving(false);
    }
  }, [fetchSettings]);

  return { settings, loading, saving, error, updateSettings, refresh: fetchSettings };
}

export function useRecallSettings() {
  const { isAuthenticated, authRequired, isLoading: authLoading } = useAuth();
  const [settings, setSettings] = useState<RecallSettings | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<ApiError | null>(null);

  const fetchSettings = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await api.getRecallSettings();
      setSettings(res);
    } catch (err) {
      setError(err as ApiError);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (authLoading) return;
    if (authRequired && !isAuthenticated) return;
    fetchSettings();
  }, [fetchSettings, authLoading, authRequired, isAuthenticated]);

  const updateSettings = useCallback(async (updates: Partial<RecallSettings>) => {
    setSaving(true);
    setError(null);
    try {
      await api.updateRecallSettings(updates);
      // Refresh settings after save
      await fetchSettings();
      return true;
    } catch (err) {
      setError(err as ApiError);
      return false;
    } finally {
      setSaving(false);
    }
  }, [fetchSettings]);

  return { settings, loading, saving, error, updateSettings, refresh: fetchSettings };
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
