/**
 * Multi-server storage system for AgentX
 * Stores configurations and metadata per-server in localStorage
 */

// === Types ===

export interface ServerConfig {
  id: string;
  name: string;
  url: string;
  isActive: boolean;
  createdAt: string;
  lastConnected?: string;
}

export interface ServerApiKeys {
  openai?: string;
  anthropic?: string;
  ollama?: string; // Ollama base URL if not default
}

export interface ServerPreferences {
  defaultModel?: string;
  defaultReasoningStrategy?: string;
  defaultDraftingStrategy?: string;
  theme?: 'dark' | 'light' | 'system';
}

export interface ServerCache {
  availableModels?: string[];
  mcpServers?: string[];
  lastHealthCheck?: string;
  lastHealthStatus?: 'healthy' | 'degraded' | 'unhealthy';
}

// === Chat Types ===

export interface ChatMessage {
  id: string;
  content: string;
  sender: 'user' | 'assistant';
  timestamp: string;
  thinking?: string;
  tokensUsed?: number;
  model?: string;
}

export interface RecentChat {
  id: string;
  sessionId: string;
  title: string;
  preview: string;
  messages: ChatMessage[];
  model?: string;
  createdAt: string;
  lastMessageAt: string;
}

export interface ServerMetadata {
  serverId: string;
  apiKeys: ServerApiKeys;
  preferences: ServerPreferences;
  cache: ServerCache;
}

// === Storage Keys ===

const STORAGE_KEYS = {
  servers: 'agentx:servers',
  activeServer: 'agentx:activeServer',
  serverMeta: (id: string) => `agentx:server:${id}:meta`,
  recentChats: (serverId: string) => `agentx:server:${serverId}:recentChats`,
} as const;

const MAX_RECENT_CHATS = 10;

// === Helper Functions ===

function generateId(): string {
  return `server_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
}

function safeJsonParse<T>(value: string | null, fallback: T): T {
  if (!value) return fallback;
  try {
    return JSON.parse(value) as T;
  } catch {
    return fallback;
  }
}

// === Server Management ===

export function getServers(): ServerConfig[] {
  const data = localStorage.getItem(STORAGE_KEYS.servers);
  return safeJsonParse<ServerConfig[]>(data, []);
}

export function saveServers(servers: ServerConfig[]): void {
  localStorage.setItem(STORAGE_KEYS.servers, JSON.stringify(servers));
}

export function addServer(name: string, url: string): ServerConfig {
  const servers = getServers();
  const isFirst = servers.length === 0;
  
  const newServer: ServerConfig = {
    id: generateId(),
    name,
    url: url.replace(/\/$/, ''), // Remove trailing slash
    isActive: isFirst, // First server is automatically active
    createdAt: new Date().toISOString(),
  };
  
  servers.push(newServer);
  saveServers(servers);
  
  // Initialize empty metadata for this server
  saveServerMetadata({
    serverId: newServer.id,
    apiKeys: {},
    preferences: {},
    cache: {},
  });
  
  if (isFirst) {
    setActiveServerId(newServer.id);
  }
  
  return newServer;
}

export function updateServer(id: string, updates: Partial<Omit<ServerConfig, 'id' | 'createdAt'>>): ServerConfig | null {
  const servers = getServers();
  const index = servers.findIndex(s => s.id === id);
  
  if (index === -1) return null;
  
  servers[index] = { ...servers[index], ...updates };
  saveServers(servers);
  
  return servers[index];
}

export function removeServer(id: string): boolean {
  const servers = getServers();
  const filtered = servers.filter(s => s.id !== id);
  
  if (filtered.length === servers.length) return false;
  
  saveServers(filtered);
  
  // Remove metadata
  localStorage.removeItem(STORAGE_KEYS.serverMeta(id));
  
  // If this was the active server, switch to another
  if (getActiveServerId() === id && filtered.length > 0) {
    setActiveServerId(filtered[0].id);
  }
  
  return true;
}

// === Active Server ===

export function getActiveServerId(): string | null {
  return localStorage.getItem(STORAGE_KEYS.activeServer);
}

export function setActiveServerId(id: string): void {
  const servers = getServers();
  
  // Update isActive flags
  const updated = servers.map(s => ({
    ...s,
    isActive: s.id === id,
  }));
  
  saveServers(updated);
  localStorage.setItem(STORAGE_KEYS.activeServer, id);
}

export function getActiveServer(): ServerConfig | null {
  const activeId = getActiveServerId();
  if (!activeId) return null;
  
  const servers = getServers();
  return servers.find(s => s.id === activeId) ?? null;
}

// === Server Metadata ===

export function getServerMetadata(serverId: string): ServerMetadata {
  const data = localStorage.getItem(STORAGE_KEYS.serverMeta(serverId));
  return safeJsonParse<ServerMetadata>(data, {
    serverId,
    apiKeys: {},
    preferences: {},
    cache: {},
  });
}

export function saveServerMetadata(metadata: ServerMetadata): void {
  localStorage.setItem(
    STORAGE_KEYS.serverMeta(metadata.serverId),
    JSON.stringify(metadata)
  );
}

export function updateServerMetadata(
  serverId: string,
  updates: Partial<Omit<ServerMetadata, 'serverId'>>
): ServerMetadata {
  const current = getServerMetadata(serverId);
  const updated: ServerMetadata = {
    ...current,
    apiKeys: { ...current.apiKeys, ...updates.apiKeys },
    preferences: { ...current.preferences, ...updates.preferences },
    cache: { ...current.cache, ...updates.cache },
  };
  
  saveServerMetadata(updated);
  return updated;
}

// === Convenience Functions ===

export function getActiveServerMetadata(): ServerMetadata | null {
  const activeId = getActiveServerId();
  if (!activeId) return null;
  return getServerMetadata(activeId);
}

export function updateActiveServerMetadata(
  updates: Partial<Omit<ServerMetadata, 'serverId'>>
): ServerMetadata | null {
  const activeId = getActiveServerId();
  if (!activeId) return null;
  return updateServerMetadata(activeId, updates);
}

export function markServerConnected(serverId: string): void {
  updateServer(serverId, { lastConnected: new Date().toISOString() });
}

// === Initialize Default Server ===

export function ensureDefaultServer(): ServerConfig {
  const servers = getServers();

  if (servers.length === 0) {
    return addServer('Local Development', 'http://localhost:12319');
  }

  // Ensure there's an active server
  const activeId = getActiveServerId();
  if (!activeId || !servers.find(s => s.id === activeId)) {
    setActiveServerId(servers[0].id);
    return servers[0];
  }

  return servers.find(s => s.id === activeId)!;
}

// === Recent Chats ===

export function getRecentChats(serverId?: string): RecentChat[] {
  const id = serverId ?? getActiveServerId();
  if (!id) return [];

  const data = localStorage.getItem(STORAGE_KEYS.recentChats(id));
  return safeJsonParse<RecentChat[]>(data, []);
}

export function saveRecentChats(chats: RecentChat[], serverId?: string): void {
  const id = serverId ?? getActiveServerId();
  if (!id) return;

  // Keep only the most recent chats
  const trimmed = chats.slice(0, MAX_RECENT_CHATS);
  localStorage.setItem(STORAGE_KEYS.recentChats(id), JSON.stringify(trimmed));
}

export function addRecentChat(chat: Omit<RecentChat, 'id'>): RecentChat {
  const serverId = getActiveServerId();
  if (!serverId) throw new Error('No active server');

  const chats = getRecentChats(serverId);

  // Generate ID
  const newChat: RecentChat = {
    ...chat,
    id: `chat_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
  };

  // Check if this sessionId already exists, update it
  const existingIndex = chats.findIndex(c => c.sessionId === chat.sessionId);
  if (existingIndex !== -1) {
    chats[existingIndex] = { ...newChat, id: chats[existingIndex].id };
  } else {
    chats.unshift(newChat);
  }

  saveRecentChats(chats, serverId);
  return existingIndex !== -1 ? chats[existingIndex] : newChat;
}

export function updateRecentChat(
  chatId: string,
  updates: Partial<Omit<RecentChat, 'id'>>
): RecentChat | null {
  const serverId = getActiveServerId();
  if (!serverId) return null;

  const chats = getRecentChats(serverId);
  const index = chats.findIndex(c => c.id === chatId);
  if (index === -1) return null;

  chats[index] = { ...chats[index], ...updates };

  // Move to top of list
  const [updated] = chats.splice(index, 1);
  chats.unshift(updated);

  saveRecentChats(chats, serverId);
  return updated;
}

export function deleteRecentChat(chatId: string): boolean {
  const serverId = getActiveServerId();
  if (!serverId) return false;

  const chats = getRecentChats(serverId);
  const filtered = chats.filter(c => c.id !== chatId);

  if (filtered.length === chats.length) return false;

  saveRecentChats(filtered, serverId);
  return true;
}

export function getRecentChatBySessionId(sessionId: string): RecentChat | null {
  const chats = getRecentChats();
  return chats.find(c => c.sessionId === sessionId) ?? null;
}
