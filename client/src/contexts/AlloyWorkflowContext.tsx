/**
 * Alloy Workflow Context — manages multi-agent workflow definitions
 * for the active server. Mirrors AgentProfileContext.
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
import {
  api,
  type AlloyWorkflow,
  type AlloyWorkflowCreate,
  type AlloyWorkflowUpdate,
} from '../lib/api';

const STORAGE_KEY_WORKFLOWS = (serverId: string) => `agentx:server:${serverId}:alloyWorkflows`;

interface AlloyWorkflowContextValue {
  workflows: AlloyWorkflow[];
  loading: boolean;
  error: string | null;

  refresh: () => Promise<void>;
  createWorkflow: (workflow: AlloyWorkflowCreate) => Promise<AlloyWorkflow>;
  updateWorkflow: (id: string, patch: AlloyWorkflowUpdate) => Promise<AlloyWorkflow>;
  deleteWorkflow: (id: string) => Promise<boolean>;
  getWorkflowById: (id: string) => AlloyWorkflow | null;
}

const AlloyWorkflowContext = createContext<AlloyWorkflowContextValue | null>(null);

export function AlloyWorkflowProvider({ children }: { children: ReactNode }) {
  const { activeServer } = useServer();
  const [workflows, setWorkflows] = useState<AlloyWorkflow[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const initializedServerId = useRef<string | null>(null);

  const load = useCallback(async () => {
    if (!activeServer) {
      setWorkflows([]);
      setLoading(false);
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const result = await api.listAlloyWorkflows();
      setWorkflows(result.workflows);
      localStorage.setItem(
        STORAGE_KEY_WORKFLOWS(activeServer.id),
        JSON.stringify(result.workflows)
      );
    } catch (err) {
      const cached = localStorage.getItem(STORAGE_KEY_WORKFLOWS(activeServer.id));
      if (cached) {
        try {
          setWorkflows(JSON.parse(cached) as AlloyWorkflow[]);
        } catch {
          setWorkflows([]);
        }
      }
      setError(err instanceof Error ? err.message : 'Failed to load workflows');
    } finally {
      setLoading(false);
    }
  }, [activeServer]);

  useEffect(() => {
    if (!activeServer) {
      setWorkflows([]);
      initializedServerId.current = null;
      setLoading(false);
      return;
    }

    if (initializedServerId.current === activeServer.id) return;
    initializedServerId.current = activeServer.id;
    load();
  }, [activeServer, load]);

  const createWorkflow = useCallback(async (workflow: AlloyWorkflowCreate) => {
    const result = await api.createAlloyWorkflow(workflow);
    await load();
    return result.workflow;
  }, [load]);

  const updateWorkflow = useCallback(async (id: string, patch: AlloyWorkflowUpdate) => {
    const result = await api.updateAlloyWorkflow(id, patch);
    await load();
    return result.workflow;
  }, [load]);

  const deleteWorkflow = useCallback(async (id: string): Promise<boolean> => {
    try {
      const result = await api.deleteAlloyWorkflow(id);
      if (result.deleted) {
        await load();
        return true;
      }
      return false;
    } catch {
      return false;
    }
  }, [load]);

  const getWorkflowById = useCallback((id: string) => {
    return workflows.find(w => w.id === id) ?? null;
  }, [workflows]);

  return (
    <AlloyWorkflowContext.Provider
      value={{
        workflows,
        loading,
        error,
        refresh: load,
        createWorkflow,
        updateWorkflow,
        deleteWorkflow,
        getWorkflowById,
      }}
    >
      {children}
    </AlloyWorkflowContext.Provider>
  );
}

export function useAlloyWorkflow() {
  const context = useContext(AlloyWorkflowContext);
  if (!context) {
    throw new Error('useAlloyWorkflow must be used within an AlloyWorkflowProvider');
  }
  return context;
}
