/**
 * Auth Context - provides authentication state and methods to all components
 */

import { createContext, useContext, useState, useEffect, useCallback, ReactNode } from 'react';
import { api, AuthStatusResponse, AuthSessionResponse } from '../lib/api';
import { getAuthToken, saveAuthToken, clearAuthToken } from '../lib/storage';

interface AuthContextValue {
  // Auth state
  isAuthenticated: boolean;
  isLoading: boolean;
  authRequired: boolean;
  setupRequired: boolean;
  sessionInfo: AuthSessionResponse | null;

  // Auth methods
  login: (username: string, password: string) => Promise<boolean>;
  logout: () => Promise<void>;
  setupPassword: (password: string, confirmPassword: string) => Promise<boolean>;
  changePassword: (oldPassword: string, newPassword: string) => Promise<boolean>;
  checkAuthStatus: () => Promise<void>;
}

const AuthContext = createContext<AuthContextValue | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [authRequired, setAuthRequired] = useState(false);
  const [setupRequired, setSetupRequired] = useState(false);
  const [sessionInfo, setSessionInfo] = useState<AuthSessionResponse | null>(null);

  // Check auth status on mount and when server changes
  const checkAuthStatus = useCallback(async () => {
    setIsLoading(true);
    try {
      // Check server's auth requirements
      const status: AuthStatusResponse = await api.authStatus();
      setAuthRequired(status.auth_required);
      setSetupRequired(status.setup_required);

      // If auth not required (disabled or bypass active), we're "authenticated"
      if (!status.auth_required) {
        setIsAuthenticated(true);
        setSessionInfo(null);
        setIsLoading(false);
        return;
      }

      // If setup required, not authenticated yet
      if (status.setup_required) {
        setIsAuthenticated(false);
        setSessionInfo(null);
        setIsLoading(false);
        return;
      }

      // Check if we have a valid token
      const token = getAuthToken();
      if (!token) {
        setIsAuthenticated(false);
        setSessionInfo(null);
        setIsLoading(false);
        return;
      }

      // Validate the token by fetching session info
      try {
        const session = await api.authSession();
        setIsAuthenticated(true);
        setSessionInfo(session);
      } catch {
        // Token invalid, clear it
        clearAuthToken();
        setIsAuthenticated(false);
        setSessionInfo(null);
      }
    } catch (error) {
      // Server unreachable - assume no auth required for now
      console.error('Failed to check auth status:', error);
      setAuthRequired(false);
      setIsAuthenticated(true);
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Initial check
  useEffect(() => {
    checkAuthStatus();
  }, [checkAuthStatus]);

  // Re-check when active server changes
  useEffect(() => {
    const handleStorageChange = (e: StorageEvent) => {
      if (e.key === 'agentx:activeServer') {
        checkAuthStatus();
      }
    };
    window.addEventListener('storage', handleStorageChange);
    return () => window.removeEventListener('storage', handleStorageChange);
  }, [checkAuthStatus]);

  // Listen for auth-required events (from 401 responses)
  useEffect(() => {
    const handleAuthRequired = () => {
      setIsAuthenticated(false);
      setSessionInfo(null);
    };
    window.addEventListener('agentx:auth-required', handleAuthRequired);
    return () => window.removeEventListener('agentx:auth-required', handleAuthRequired);
  }, []);

  const login = useCallback(async (username: string, password: string): Promise<boolean> => {
    try {
      const response = await api.login({ username, password });
      saveAuthToken(response.token);
      setIsAuthenticated(true);

      // Fetch session info
      try {
        const session = await api.authSession();
        setSessionInfo(session);
      } catch {
        // Session fetch failed but login succeeded
      }

      return true;
    } catch (error) {
      console.error('Login failed:', error);
      return false;
    }
  }, []);

  const logout = useCallback(async (): Promise<void> => {
    try {
      await api.logout();
    } catch {
      // Logout request failed, but we'll clear local state anyway
    }
    clearAuthToken();
    setIsAuthenticated(false);
    setSessionInfo(null);
  }, []);

  const setupPassword = useCallback(async (password: string, confirmPassword: string): Promise<boolean> => {
    try {
      await api.authSetup({ password, confirm_password: confirmPassword });
      setSetupRequired(false);
      // After setup, user needs to login
      return true;
    } catch (error) {
      console.error('Password setup failed:', error);
      return false;
    }
  }, []);

  const changePassword = useCallback(async (oldPassword: string, newPassword: string): Promise<boolean> => {
    try {
      await api.changePassword({ old_password: oldPassword, new_password: newPassword });
      return true;
    } catch (error) {
      console.error('Password change failed:', error);
      return false;
    }
  }, []);

  const value: AuthContextValue = {
    isAuthenticated,
    isLoading,
    authRequired,
    setupRequired,
    sessionInfo,
    login,
    logout,
    setupPassword,
    changePassword,
    checkAuthStatus,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthContextValue {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}
