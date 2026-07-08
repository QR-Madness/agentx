/**
 * ConsolidationContext — app-wide consolidation run state.
 *
 * Lifts `useConsolidationStream` above the drawer so a run survives the drawer
 * closing, and the ⚡ TopBar pulse + the drawer read one source. Adds a
 * visibility-aware poll of `/api/jobs` while idle so a run started elsewhere
 * (the background worker, another tab) is picked up and watched — gated off
 * while a stream is already active and while the tab is hidden.
 */

import { createContext, useContext, useEffect, useRef, type ReactNode } from 'react';
import { api } from '../lib/api';
import { useConsolidationStream } from '../lib/hooks';

type ConsolidationValue = ReturnType<typeof useConsolidationStream> & {
  isActive: boolean;
  trigger: (jobs?: string[]) => { abort: () => void } | undefined;
};

const ConsolidationContext = createContext<ConsolidationValue | null>(null);

const POLL_MS = 5000;

export function ConsolidationProvider({ children }: { children: ReactNode }) {
  const stream = useConsolidationStream();
  // Latest streaming flag for the poller without re-arming the interval each tick.
  const streamingRef = useRef(stream.isStreaming);
  streamingRef.current = stream.isStreaming;
  const startRef = useRef(stream.startStream);
  startRef.current = stream.startStream;

  useEffect(() => {
    let cancelled = false;

    async function poll() {
      if (cancelled || streamingRef.current || document.visibilityState !== 'visible') return;
      try {
        const jobs = await api.listJobs();
        if (cancelled || streamingRef.current) return;
        if (jobs.consolidation_active) {
          startRef.current({ trigger: false }); // GET = watch-only
        }
      } catch {
        // Server may be unavailable — ignore, try again next tick.
      }
    }

    poll(); // once on mount
    const id = window.setInterval(poll, POLL_MS);
    const onVis = () => {
      if (document.visibilityState === 'visible') poll();
    };
    document.addEventListener('visibilitychange', onVis);
    return () => {
      cancelled = true;
      window.clearInterval(id);
      document.removeEventListener('visibilitychange', onVis);
    };
  }, []);

  const value: ConsolidationValue = {
    ...stream,
    isActive: stream.isStreaming,
    trigger: (jobs?: string[]) => stream.startStream({ trigger: true, jobs }),
  };

  return <ConsolidationContext.Provider value={value}>{children}</ConsolidationContext.Provider>;
}

export function useConsolidation(): ConsolidationValue {
  const ctx = useContext(ConsolidationContext);
  if (!ctx) throw new Error('useConsolidation must be used within ConsolidationProvider');
  return ctx;
}
