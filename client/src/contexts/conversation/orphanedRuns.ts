/**
 * Recovery-surface helper: which detached chat runs are "orphaned" — i.e. still
 * running but not owned by any open tab, so they're the ones a user would want
 * to resume. Every foreground chat also creates a detached run, so without this
 * filter the active conversation would show up as "recoverable".
 */

import type { ActiveChatRun } from '../../lib/api';
import type { ConversationTab } from '../../lib/storage';

export function orphanedRuns(
  runs: ActiveChatRun[],
  tabs: ConversationTab[],
): ActiveChatRun[] {
  const ownedRunIds = new Set(
    tabs.map(t => t.activeRun?.runId).filter((id): id is string => Boolean(id)),
  );
  return runs.filter(r => r.status === 'running' && !ownedRunIds.has(r.run_id));
}
