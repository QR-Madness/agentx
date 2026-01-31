/**
 * Server Context - provides active server configuration to all components
 */

import { createContext, useContext, useState, useEffect, useCallback, ReactNode } from 'react';
import {
  ServerConfig,
  ServerMetadata,
  getServers,
  getActiveServer,
  getActiveServerMetadata,
  setActiveServerId,
  addServer,
  updateServer,
  removeServer,
  updateActiveServerMetadata,
  ensureDefaultServer,
} from '../lib/storage';

interface ServerContextValue {
  // Current server state
  servers: ServerConfig[];
  activeServer: ServerConfig | null;
  activeMetadata: ServerMetadata | null;
  
  // Server management
  switchServer: (id: string) => void;
  addNewServer: (name: string, url: string) => ServerConfig;
  updateServerConfig: (id: string, updates: Partial<ServerConfig>) => void;
  deleteServer: (id: string) => void;
  
  // Metadata management
  updateMetadata: (updates: Partial<Omit<ServerMetadata, 'serverId'>>) => void;
  
  // Refresh
  refreshServers: () => void;
}

const ServerContext = createContext<ServerContextValue | null>(null);

export function ServerProvider({ children }: { children: ReactNode }) {
  const [servers, setServersState] = useState<ServerConfig[]>([]);
  const [activeServer, setActiveServerState] = useState<ServerConfig | null>(null);
  const [activeMetadata, setActiveMetadataState] = useState<ServerMetadata | null>(null);

  const refreshServers = useCallback(() => {
    const allServers = getServers();
    const active = getActiveServer();
    const meta = active ? getActiveServerMetadata() : null;
    
    setServersState(allServers);
    setActiveServerState(active);
    setActiveMetadataState(meta);
  }, []);

  // Initialize on mount
  useEffect(() => {
    ensureDefaultServer();
    refreshServers();
  }, [refreshServers]);

  const switchServer = useCallback((id: string) => {
    setActiveServerId(id);
    refreshServers();
  }, [refreshServers]);

  const addNewServer = useCallback((name: string, url: string) => {
    const server = addServer(name, url);
    refreshServers();
    return server;
  }, [refreshServers]);

  const updateServerConfig = useCallback((id: string, updates: Partial<ServerConfig>) => {
    updateServer(id, updates);
    refreshServers();
  }, [refreshServers]);

  const deleteServer = useCallback((id: string) => {
    removeServer(id);
    refreshServers();
  }, [refreshServers]);

  const updateMetadata = useCallback((updates: Partial<Omit<ServerMetadata, 'serverId'>>) => {
    updateActiveServerMetadata(updates);
    refreshServers();
  }, [refreshServers]);

  const value: ServerContextValue = {
    servers,
    activeServer,
    activeMetadata,
    switchServer,
    addNewServer,
    updateServerConfig,
    deleteServer,
    updateMetadata,
    refreshServers,
  };

  return (
    <ServerContext.Provider value={value}>
      {children}
    </ServerContext.Provider>
  );
}

export function useServer() {
  const context = useContext(ServerContext);
  if (!context) {
    throw new Error('useServer must be used within a ServerProvider');
  }
  return context;
}
