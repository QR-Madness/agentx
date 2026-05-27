import { describe, it, expect } from 'vitest';
import { classifyStatus, isApiError, toApiError, apiErrorMessage, type ApiError } from './api';

describe('classifyStatus', () => {
  it('maps statuses to their coarse kind', () => {
    expect(classifyStatus(400)).toBe('bad_request');
    expect(classifyStatus(401)).toBe('auth');
    expect(classifyStatus(404)).toBe('not_found');
    expect(classifyStatus(408)).toBe('network');
    expect(classifyStatus(502)).toBe('upstream');
    expect(classifyStatus(503)).toBe('unavailable');
    expect(classifyStatus(500)).toBe('server');
    expect(classifyStatus(0)).toBe('network');
  });
});

describe('isApiError', () => {
  it('recognizes the ApiError shape', () => {
    const err: ApiError = { message: 'x', status: 404, kind: 'not_found' };
    expect(isApiError(err)).toBe(true);
  });
  it('rejects plain errors and non-objects', () => {
    expect(isApiError(new Error('x'))).toBe(false);
    expect(isApiError('x')).toBe(false);
    expect(isApiError(null)).toBe(false);
  });
});

describe('toApiError', () => {
  it('passes an ApiError through unchanged', () => {
    const err: ApiError = { message: 'boom', status: 502, kind: 'upstream' };
    expect(toApiError(err)).toBe(err);
  });

  it('treats a bare string as a server error', () => {
    expect(toApiError('stream failed')).toEqual({
      message: 'stream failed',
      status: 0,
      kind: 'server',
    });
  });

  it('maps a fetch TypeError to a network error', () => {
    const te = new TypeError('Failed to fetch');
    const out = toApiError(te);
    expect(out.kind).toBe('network');
    expect(out.message).toBe('Cannot reach server');
  });

  it('falls back for unknown values', () => {
    expect(toApiError({ weird: true }).kind).toBe('server');
  });
});

describe('apiErrorMessage', () => {
  it('extracts the user-facing message from any input', () => {
    expect(apiErrorMessage({ message: 'Provider down', status: 502, kind: 'upstream' })).toBe('Provider down');
    expect(apiErrorMessage('raw string')).toBe('raw string');
    expect(apiErrorMessage(new Error('generic'))).toBe('generic');
  });
});
