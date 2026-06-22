/**
 * Ambassador relay decision — pure logic shared by the voice bar and the text composer.
 *
 * Relay only ever reaches the conversation the person is *in* (the active tab): it is the
 * one conversation with a live send handler (ChatPanel registers its send/steer path per
 * active tab). The relayed text becomes a real *user* turn — the ambassador stays a
 * non-participant; the user is the author. This helper picks the target, invokes the send
 * seam, and reports where it landed (or why it couldn't) so both surfaces give the same
 * closure instead of failing silently.
 */

export interface RelayTab {
  id: string;
  title?: string;
  activeRun?: { runId?: string | null } | null;
}

export interface RelayOutcome {
  ok: boolean;
  note: string;
}

/**
 * @param text   the message to relay
 * @param activeTab the conversation the person is in (null if none open)
 * @param send   the registered relay seam (returns false if the tab has no live handler)
 */
export function relayToActiveConversation(
  text: string,
  activeTab: RelayTab | null | undefined,
  send: (tabId: string, text: string) => boolean,
): RelayOutcome {
  const t = text.trim();
  if (!t) return { ok: false, note: 'Nothing to send.' };
  if (!activeTab) return { ok: false, note: 'Open a conversation to relay into.' };
  if (!send(activeTab.id, t)) {
    return { ok: false, note: 'Could not send to the conversation.' };
  }
  return {
    ok: true,
    note: activeTab.activeRun?.runId
      ? 'Folded into the running turn.'
      : `Sent to ${activeTab.title || 'the conversation'}.`,
  };
}
