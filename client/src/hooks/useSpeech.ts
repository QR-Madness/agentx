/**
 * useSpeech — thin React surface over the `speechPlayer` singleton (lib/audio.ts).
 *
 * Exposes per-item playback state (`playingId`/`loadingId`) and `speak`/`stop`/
 * `unlock`, wiring synthesis to the ambassador `/speak` endpoint. Speech runs
 * through one shared audio element, so a new utterance stops the prior one
 * app-wide. The component owns *when* to speak (on-demand button vs voice-mode
 * autoplay); this hook just plays.
 */

import { useCallback, useState, useSyncExternalStore } from 'react';
import { speechPlayer, type Synthesize } from '../lib/audio';
import { api } from '../lib/api';
import { apiErrorMessage } from '../lib/api';

interface UseSpeechOptions {
  /** Ambassador profile whose speech model/voice to use (default ambassador if omitted). */
  agentProfileId?: string;
}

export function useSpeech(opts: UseSpeechOptions = {}) {
  const { agentProfileId } = opts;
  const snapshot = useSyncExternalStore(speechPlayer.subscribe, speechPlayer.getSnapshot);
  const [error, setError] = useState<string | null>(null);

  const speak = useCallback(
    async (id: string, text: string) => {
      setError(null);
      const synthesize: Synthesize = (t, signal) =>
        api.speak({ text: t, agent_profile_id: agentProfileId }, signal);
      try {
        await speechPlayer.speak(id, text, synthesize);
      } catch (err) {
        // Aborts (new synth / stop) are expected — don't surface them.
        if (err instanceof DOMException && err.name === 'AbortError') return;
        setError(apiErrorMessage(err));
      }
    },
    [agentProfileId],
  );

  const stop = useCallback(() => speechPlayer.stop(), []);
  const unlock = useCallback(() => speechPlayer.unlock(), []);

  return {
    playingId: snapshot.playingId,
    loadingId: snapshot.loadingId,
    error,
    speak,
    stop,
    unlock,
  };
}
