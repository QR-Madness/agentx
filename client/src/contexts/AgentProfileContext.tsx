/**
 * Agent Profile Context — Manages agent profiles with server-specific caching
 */

import {
  createContext,
  useContext,
  useState,
  useCallback,
  useEffect,
  useRef,
  type ReactNode,
} from 'react';
import { useServer } from './ServerContext';
import { api, type AgentProfile, type AgentProfileCreate } from '../lib/api';

// Storage keys (per-server)
const STORAGE_KEY_PROFILES = (serverId: string) => `agentx:server:${serverId}:agentProfiles`;
const STORAGE_KEY_ACTIVE = (serverId: string) => `agentx:server:${serverId}:activeAgentProfile`;

interface AgentProfileContextValue {
  profiles: AgentProfile[];
  activeProfile: AgentProfile | null;
  loading: boolean;
  error: string | null;

  // Actions
  setActiveProfile: (id: string) => void;
  createProfile: (profile: AgentProfileCreate) => Promise<AgentProfile>;
  updateProfile: (id: string, updates: Partial<AgentProfileCreate>) => Promise<AgentProfile | null>;
  deleteProfile: (id: string) => Promise<boolean>;
  setDefaultProfile: (id: string) => Promise<boolean>;
  refresh: () => Promise<void>;

  // Convenience getters
  getAgentName: () => string;
}

const AgentProfileContext = createContext<AgentProfileContextValue | null>(null);

export function AgentProfileProvider({ children }: { children: ReactNode }) {
  const { activeServer } = useServer();
  const [profiles, setProfiles] = useState<AgentProfile[]>([]);
  const [activeProfileId, setActiveProfileId] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const initializedServerId = useRef<string | null>(null);

  // Load profiles when server changes
  const loadProfiles = useCallback(async () => {
    if (!activeServer) {
      setProfiles([]);
      setActiveProfileId(null);
      setLoading(false);
      return;
    }

    setLoading(true);
    setError(null);

    try {
      // Fetch from server
      const result = await api.listAgentProfiles();
      setProfiles(result.profiles);

      // Load active profile ID from localStorage
      const storedActiveId = localStorage.getItem(STORAGE_KEY_ACTIVE(activeServer.id));
      const validActiveId = result.profiles.some(p => p.id === storedActiveId)
        ? storedActiveId
        : result.profiles.find(p => p.isDefault)?.id || result.profiles[0]?.id || null;

      setActiveProfileId(validActiveId);

      // Cache profiles in localStorage
      localStorage.setItem(
        STORAGE_KEY_PROFILES(activeServer.id),
        JSON.stringify(result.profiles)
      );
    } catch (err) {
      // Fall back to cached profiles
      const cached = localStorage.getItem(STORAGE_KEY_PROFILES(activeServer.id));
      if (cached) {
        try {
          const cachedProfiles = JSON.parse(cached) as AgentProfile[];
          setProfiles(cachedProfiles);
          const storedActiveId = localStorage.getItem(STORAGE_KEY_ACTIVE(activeServer.id));
          setActiveProfileId(storedActiveId || cachedProfiles[0]?.id || null);
        } catch {
          setProfiles([]);
          setActiveProfileId(null);
        }
      }
      setError(err instanceof Error ? err.message : 'Failed to load profiles');
    } finally {
      setLoading(false);
    }
  }, [activeServer]);

  useEffect(() => {
    if (!activeServer) {
      setProfiles([]);
      setActiveProfileId(null);
      initializedServerId.current = null;
      setLoading(false);
      return;
    }

    if (initializedServerId.current === activeServer.id) {
      return;
    }

    initializedServerId.current = activeServer.id;
    loadProfiles();
  }, [activeServer, loadProfiles]);

  // Save active profile ID to localStorage
  useEffect(() => {
    if (!activeServer || !activeProfileId) return;
    localStorage.setItem(STORAGE_KEY_ACTIVE(activeServer.id), activeProfileId);
  }, [activeProfileId, activeServer]);

  const activeProfile = profiles.find(p => p.id === activeProfileId) ?? null;

  const setActiveProfile = useCallback((id: string) => {
    if (profiles.some(p => p.id === id)) {
      setActiveProfileId(id);
    }
  }, [profiles]);

  const createProfile = useCallback(async (profile: AgentProfileCreate): Promise<AgentProfile> => {
    const result = await api.createAgentProfile(profile);
    await loadProfiles();
    return result.profile;
  }, [loadProfiles]);

  const updateProfile = useCallback(async (
    id: string,
    updates: Partial<AgentProfileCreate>
  ): Promise<AgentProfile | null> => {
    try {
      const result = await api.updateAgentProfile(id, updates);
      await loadProfiles();
      return result.profile;
    } catch {
      return null;
    }
  }, [loadProfiles]);

  const deleteProfile = useCallback(async (id: string): Promise<boolean> => {
    try {
      const result = await api.deleteAgentProfile(id);
      if (result.deleted) {
        await loadProfiles();
        return true;
      }
      return false;
    } catch {
      return false;
    }
  }, [loadProfiles]);

  const setDefaultProfile = useCallback(async (id: string): Promise<boolean> => {
    try {
      await api.setDefaultAgentProfile(id);
      await loadProfiles();
      return true;
    } catch {
      return false;
    }
  }, [loadProfiles]);

  const getAgentName = useCallback((): string => {
    return activeProfile?.name ?? 'AgentX';
  }, [activeProfile]);

  return (
    <AgentProfileContext.Provider
      value={{
        profiles,
        activeProfile,
        loading,
        error,
        setActiveProfile,
        createProfile,
        updateProfile,
        deleteProfile,
        setDefaultProfile,
        refresh: loadProfiles,
        getAgentName,
      }}
    >
      {children}
    </AgentProfileContext.Provider>
  );
}

export function useAgentProfile() {
  const context = useContext(AgentProfileContext);
  if (!context) {
    throw new Error('useAgentProfile must be used within an AgentProfileProvider');
  }
  return context;
}
