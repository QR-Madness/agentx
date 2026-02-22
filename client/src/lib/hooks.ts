/**
 * React hooks for API data fetching
 */

import { useState, useEffect, useCallback } from 'react';
import {
  api,
  HealthResponse,
  ProviderInfo,
  MCPServer,
  MCPTool,
  ApiError,
  MemoryChannel,
  MemoryEntity,
  MemoryFact,
  MemoryStrategy,
  MemoryStats,
  EntityGraph
} from './api';

export function useHealth(includeMemory = true) {
  const [data, setData] = useState<HealthResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<ApiError | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const result = await api.health(includeMemory);
      setData(result);
    } catch (err) {
      setError(err as ApiError);
    } finally {
      setLoading(false);
    }
  }, [includeMemory]);

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

  useEffect(() => {
    refresh();
  }, [refresh]);

  return { servers, loading, error, refresh };
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
