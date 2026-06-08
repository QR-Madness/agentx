/**
 * useDictation — push-to-talk voice input for the ambassador (the STT half of
 * voice mode). Captures mic audio via `AudioRecorder`, transcribes it through the
 * ambassador `/transcribe` endpoint, and hands the text back to the caller to
 * place into the reviewable input — it **never** auto-sends (pre-send confirmation).
 *
 * State machine: `idle → recording → transcribing → idle`. The caller owns the
 * trigger (a record button, hold-space, or the future floating CC player).
 */

import { useCallback, useRef, useState } from 'react';
import {
  AudioRecorder,
  RecordingError,
  blobToBase64,
  isRecordingSupported,
} from '../lib/audioRecorder';
import { api, apiErrorMessage } from '../lib/api';

export type DictationState = 'idle' | 'recording' | 'transcribing';

interface UseDictationOptions {
  /** Ambassador profile whose STT model to use (default ambassador if omitted). */
  agentProfileId?: string;
  /** Receives the transcript so the caller can place it into the input for review. */
  onTranscript: (text: string) => void;
}

export function useDictation({ agentProfileId, onTranscript }: UseDictationOptions) {
  const recorderRef = useRef<AudioRecorder | null>(null);
  const [state, setState] = useState<DictationState>('idle');
  const [error, setError] = useState<string | null>(null);

  const getRecorder = () => (recorderRef.current ??= new AudioRecorder());

  const start = useCallback(async () => {
    setError(null);
    setState((cur) => (cur === 'idle' ? 'recording' : cur));
    try {
      await getRecorder().start();
    } catch (err) {
      setState('idle');
      setError(err instanceof RecordingError ? err.message : 'Could not start recording.');
    }
  }, []);

  const stopAndTranscribe = useCallback(async () => {
    const recorder = recorderRef.current;
    if (!recorder) {
      setState('idle');
      return;
    }
    setState('transcribing');
    let result;
    try {
      result = await recorder.stop();
    } catch {
      result = null;
    }
    if (!result || result.blob.size < 512) {
      // Empty/near-empty clip — usually the mic captured nothing or (on Linux) the
      // GStreamer audio codecs aren't installed. Surface it rather than no-op.
      setError('No audio was captured — check your microphone (and, on Linux, the GStreamer codecs).');
      setState('idle');
      return;
    }
    try {
      const audio = await blobToBase64(result.blob);
      const { text } = await api.transcribe({
        audio,
        format: result.format,
        agent_profile_id: agentProfileId,
      });
      const trimmed = text.trim();
      if (trimmed) onTranscript(trimmed);
    } catch (err) {
      setError(apiErrorMessage(err));
    } finally {
      setState('idle');
    }
  }, [agentProfileId, onTranscript]);

  /** Abandon an in-progress recording without transcribing (retake / bail). */
  const cancel = useCallback(() => {
    recorderRef.current?.cancel();
    setState('idle');
  }, []);

  return {
    state,
    error,
    supported: isRecordingSupported(),
    recording: state === 'recording',
    transcribing: state === 'transcribing',
    start,
    stopAndTranscribe,
    cancel,
  };
}
