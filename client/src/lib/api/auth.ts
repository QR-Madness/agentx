import { request as apiRequest } from './core';
import type { AuthChangePasswordRequest, AuthLoginRequest, AuthLoginResponse, AuthSessionResponse, AuthSetupRequest, AuthStatusResponse } from './types';

export const authApi = {
  // === Authentication ===

  async authStatus(): Promise<AuthStatusResponse> {
    return apiRequest<AuthStatusResponse>('/api/auth/status', {}, true);
  },

  async login(credentials: AuthLoginRequest): Promise<AuthLoginResponse> {
    return apiRequest<AuthLoginResponse>('/api/auth/login', {
      method: 'POST',
      body: JSON.stringify(credentials),
    }, true);
  },

  async logout(): Promise<{ message: string }> {
    return apiRequest<{ message: string }>('/api/auth/logout', {
      method: 'POST',
    });
  },

  async authSession(): Promise<AuthSessionResponse> {
    return apiRequest<AuthSessionResponse>('/api/auth/session');
  },

  async authSetup(data: AuthSetupRequest): Promise<{ message: string }> {
    return apiRequest<{ message: string }>('/api/auth/setup', {
      method: 'POST',
      body: JSON.stringify(data),
    }, true);
  },

  async changePassword(data: AuthChangePasswordRequest): Promise<{ message: string }> {
    return apiRequest<{ message: string }>('/api/auth/change-password', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  },
};
