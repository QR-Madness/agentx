/**
 * React hooks for API data fetching
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import { useAuth } from '../contexts/AuthContext';
import {
  api,
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
  MemoryStrategy,
  MemoryStats,
  EntityGraph,
  JobStatus,
  JobHistory,
  WorkerStatus,
  ConsolidateResult,
  ConsolidationSettings,
  RecallSettings,
} from './api';

export function useHealth(includeMemory = true, includeStorage = true) {
  const [data, setData] = useState<HealthResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<ApiError | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const result = await api.health(includeMemory, includeStorage);
      setData(result);
    } catch (err) {
      setError(err as ApiError);
    } finally {
      setLoading(false);
    }
  }, [includeMemory, includeStorage]);

  useEffect(() => {
    refresh();
  }, [refresh]);

  return { data, loading, error, refresh };
}

export function useProviders() {
  const [providers, setProviders] = useState<ProviderInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<ApiError | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const result = await api.listProviders();
      setProviders(result.providers || []);
    } catch (err) {
      setError(err as ApiError);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  return { providers, loading, error, refresh };
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
  const [tools, setTools] = useState<MCPTool[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<ApiError | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const result = await api.listMCPTools();
      setTools(result.tools || []);
    } catch (err) {
      setError(err as ApiError);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  return { tools, loading, error, refresh };
}

export function useAgentStatus() {
  const [status, setStatus] = useState<{ status: string; active_sessions: number } | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<ApiError | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const result = await api.getAgentStatus();
      setStatus(result);
    } catch (err) {
      setError(err as ApiError);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  return { status, loading, error, refresh };
}

// === Memory Hooks ===

export function useMemoryChannels() {
  const [channels, setChannels] = useState<MemoryChannel[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<ApiError | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const result = await api.listMemoryChannels();
      setChannels(result.channels || []);
    } catch (err) {
      setError(err as ApiError);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  return { channels, loading, error, refresh };
}

export function useMemoryEntities(
  channel: string,
  page: number,
  search?: string,
  type?: string
) {
  const [entities, setEntities] = useState<MemoryEntity[]>([]);
  const [total, setTotal] = useState(0);
  const [hasNext, setHasNext] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<ApiError | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const result = await api.listMemoryEntities({
        channel,
        page,
        search: search || undefined,
        type: type || undefined
      });
      setEntities(result.entities || []);
      setTotal(result.total || 0);
      setHasNext(result.has_next || false);
    } catch (err) {
      setError(err as ApiError);
    } finally {
      setLoading(false);
    }
  }, [channel, page, search, type]);

  useEffect(() => {
    refresh();
  }, [refresh]);

  return { entities, total, hasNext, loading, error, refresh };
}

export function useEntityGraph(entityId: string | null) {
  const [graph, setGraph] = useState<EntityGraph | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<ApiError | null>(null);

  const refresh = useCallback(async () => {
    if (!entityId) {
      setGraph(null);
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const result = await api.getEntityGraph(entityId);
      setGraph(result);
    } catch (err) {
      setError(err as ApiError);
    } finally {
      setLoading(false);
    }
  }, [entityId]);

  useEffect(() => {
    refresh();
  }, [refresh]);

  return { graph, loading, error, refresh };
}

export function useMemoryFacts(
  channel: string,
  page: number,
  minConfidence?: number,
  search?: string
) {
  const [facts, setFacts] = useState<MemoryFact[]>([]);
  const [total, setTotal] = useState(0);
  const [hasNext, setHasNext] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<ApiError | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const result = await api.listMemoryFacts({
        channel,
        page,
        min_confidence: minConfidence,
        search: search || undefined
      });
      setFacts(result.facts || []);
      setTotal(result.total || 0);
      setHasNext(result.has_next || false);
    } catch (err) {
      setError(err as ApiError);
    } finally {
      setLoading(false);
    }
  }, [channel, page, minConfidence, search]);

  useEffect(() => {
    refresh();
  }, [refresh]);

  return { facts, total, hasNext, loading, error, refresh };
}

export function useMemoryStrategies(channel: string, page: number) {
  const [strategies, setStrategies] = useState<MemoryStrategy[]>([]);
  const [total, setTotal] = useState(0);
  const [hasNext, setHasNext] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<ApiError | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const result = await api.listMemoryStrategies({ channel, page });
      setStrategies(result.strategies || []);
      setTotal(result.total || 0);
      setHasNext(result.has_next || false);
    } catch (err) {
      setError(err as ApiError);
    } finally {
      setLoading(false);
    }
  }, [channel, page]);

  useEffect(() => {
    refresh();
  }, [refresh]);

  return { strategies, total, hasNext, loading, error, refresh };
}

export function useMemoryStats() {
  const [stats, setStats] = useState<MemoryStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<ApiError | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const result = await api.getMemoryStats();
      setStats(result);
    } catch (err) {
      setError(err as ApiError);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  return { stats, loading, error, refresh };
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
  const [jobs, setJobs] = useState<JobStatus[]>([]);
  const [worker, setWorker] = useState<WorkerStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<ApiError | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const result = await api.listJobs();
      setJobs(result.jobs);
      setWorker(result.worker);
    } catch (err) {
      setError(err as ApiError);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  return { jobs, worker, loading, error, refresh };
}

export function useJob(name: string) {
  const [job, setJob] = useState<JobStatus | null>(null);
  const [history, setHistory] = useState<JobHistory[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<ApiError | null>(null);

  const refresh = useCallback(async () => {
    if (!name) return;
    setLoading(true);
    setError(null);
    try {
      const result = await api.getJob(name);
      setJob(result.job);
      setHistory(result.history);
    } catch (err) {
      setError(err as ApiError);
    } finally {
      setLoading(false);
    }
  }, [name]);

  useEffect(() => {
    refresh();
  }, [refresh]);

  return { job, history, loading, error, refresh };
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
