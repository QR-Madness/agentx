/**
 * A failed turn whose stream `error` carries a `code` the client knows offers
 * its fix in Settings from the error toast. Driven through the production
 * wiring, as the golden-stream suite is: `useChatStream.attach()` → the real
 * `dispatchSseEvent` → `onError` → the real NotificationProvider.
 */

import { describe, it, expect, vi, afterEach } from 'vitest';
import { act, renderHook } from '@testing-library/react';
import type { ReactNode } from 'react';

import { useChatStream } from './useChatStream';
import { streamErrorFix } from './streamErrorFixes';
import { NotificationProvider, useNotify } from '../../contexts/NotificationContext';
import { api } from '../../lib/api';
import { dispatchSseEvent, type StreamCallbacks } from '../../lib/api/streaming';
import type { ConversationMessage } from '../../lib/messages';

/** The server's message, verbatim (agent.core.NO_MODEL_CHOSEN_MESSAGE). */
const NO_MODEL =
  'No model chosen: set the Global Default Model in Settings → Model Roles, ' +
  'or pick a model for this conversation.';

afterEach(() => {
  vi.restoreAllMocks();
});

/** Attach to a run, then end it with an `error` event carrying `data`. */
function failTurn(
  data: Record<string, unknown>,
  onOpenSettings?: (section: string, setting?: string) => void,
) {
  const messages: ConversationMessage[] = [];
  let callbacks: StreamCallbacks | null = null;
  vi.spyOn(api, 'attachChatRun').mockImplementation((_runId: string, cbs: StreamCallbacks) => {
    callbacks = cbs;
    return { abort: () => {} };
  });

  const wrapper = ({ children }: { children: ReactNode }) => (
    <NotificationProvider>{children}</NotificationProvider>
  );
  const hook = renderHook(
    () => ({
      stream: useChatStream({
        appendMessage: (m) => { messages.push(m); },
        updateMessage: () => {},
        onOpenSettings,
      }),
      notifications: useNotify(),
    }),
    { wrapper },
  );

  act(() => { hook.result.current.stream.attach('run-under-test'); });
  act(() => { dispatchSseEvent('error', data, callbacks!, new AbortController()); });

  const toasts = hook.result.current.notifications.toasts;
  expect(toasts).toHaveLength(1);
  /** Fail the next attached run the same way, as a retry would. */
  const failAgain = (next: Record<string, unknown>) => {
    act(() => { hook.result.current.stream.attach('run-retry'); });
    act(() => { dispatchSseEvent('error', next, callbacks!, new AbortController()); });
    return hook.result.current.notifications.toasts;
  };
  return { toast: toasts[0], messages, failAgain };
}

describe('chat error → settings fix', () => {
  it('offers Open Model Roles, landing on the Global Default Model, when no model is chosen', () => {
    const onOpenSettings = vi.fn();
    const { toast, messages } = failTurn({ error: NO_MODEL, code: 'no_model_chosen' }, onOpenSettings);

    expect(toast).toMatchObject({ kind: 'error', title: 'No model chosen', message: NO_MODEL });
    // Stays until acted on or dismissed: every turn fails until this is fixed.
    expect(toast.duration).toBe(0);
    expect(toast.action?.label).toBe('Open Model Roles');
    act(() => { toast.action!.onClick(); });
    expect(onOpenSettings).toHaveBeenCalledWith('model-roles', 'config:preferences.default_model');

    // The transcript keeps the server's message, which names the same place.
    expect(messages.at(-1)).toMatchObject({
      type: 'assistant',
      content: `Sorry, I encountered an error: ${NO_MODEL}`,
    });
  });

  it('offers Open Model Providers when no provider is connected yet', () => {
    const onOpenSettings = vi.fn();
    const { toast } = failTurn(
      { error: 'No model provider connected: …', code: 'no_provider_connected' },
      onOpenSettings,
    );
    expect(toast.title).toBe('No model provider connected');
    expect(toast.action?.label).toBe('Open Model Providers');
    act(() => { toast.action!.onClick(); });
    expect(onOpenSettings).toHaveBeenCalledWith('providers', undefined);
  });

  it('keeps the plain toast for an error without a code', () => {
    const onOpenSettings = vi.fn();
    const { toast } = failTurn({ error: "Provider 'lmstudio' not configured" }, onOpenSettings);
    expect(toast.kind).toBe('error');
    expect(toast.message).toBe("Provider 'lmstudio' not configured");
    expect(toast.action).toBeUndefined();
    // A plain error keeps the normal timed toast.
    expect(toast.duration).toBeGreaterThan(0);
    expect(onOpenSettings).not.toHaveBeenCalled();
  });

  it('ignores a code it does not know, and an empty one', () => {
    expect(failTurn({ error: 'slow down', code: 'rate_limited' }, vi.fn()).toast.action)
      .toBeUndefined();
    vi.restoreAllMocks();
    expect(failTurn({ error: 'odd', code: '' }, vi.fn()).toast.action).toBeUndefined();
  });

  it('replaces its toast on a retry rather than stacking persistent ones', () => {
    const { toast, failAgain } = failTurn({ error: NO_MODEL, code: 'no_model_chosen' }, vi.fn());
    const after = failAgain({ error: NO_MODEL, code: 'no_model_chosen' });
    expect(after).toHaveLength(1);
    expect(after[0].id).not.toBe(toast.id);
    expect(after[0].action?.label).toBe('Open Model Roles');
  });

  it('offers no button where Settings cannot be opened', () => {
    const { toast } = failTurn({ error: NO_MODEL, code: 'no_model_chosen' });
    expect(toast.action).toBeUndefined();
    expect(toast.message).toBe(NO_MODEL);
  });
});

describe('streamErrorFix', () => {
  it('knows only its own codes', () => {
    expect(streamErrorFix('no_model_chosen')?.section).toBe('model-roles');
    expect(streamErrorFix(undefined)).toBeUndefined();
    // An inherited property name is not a code.
    expect(streamErrorFix('toString')).toBeUndefined();
  });
});
