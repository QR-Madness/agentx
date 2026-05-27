import { describe, it, expect, vi } from 'vitest';
import { renderHook, waitFor } from '@testing-library/react';
import { useApi } from './hooks';
import type { ApiError } from './api';

describe('useApi', () => {
  it('resolves data and clears loading', async () => {
    const { result } = renderHook(() => useApi(() => Promise.resolve({ value: 42 }), []));

    expect(result.current.loading).toBe(true);
    await waitFor(() => expect(result.current.loading).toBe(false));

    expect(result.current.data).toEqual({ value: 42 });
    expect(result.current.error).toBeNull();
  });

  it('captures a normalized ApiError on failure', async () => {
    const err: ApiError = { message: 'nope', status: 404, kind: 'not_found' };
    const { result } = renderHook(() => useApi(() => Promise.reject(err), []));

    await waitFor(() => expect(result.current.loading).toBe(false));

    expect(result.current.error?.kind).toBe('not_found');
    expect(result.current.error?.message).toBe('nope');
    expect(result.current.data).toBeNull();
  });

  it('does not fetch when disabled', async () => {
    const call = vi.fn(() => Promise.resolve(1));
    const { result } = renderHook(() => useApi(call, [], { enabled: false }));

    expect(result.current.loading).toBe(false);
    expect(call).not.toHaveBeenCalled();
  });
});
