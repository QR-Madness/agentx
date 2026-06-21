/**
 * WorkspaceContainerCard — resource + lifecycle for a container-backed workspace shell.
 *
 * Shows live state/mem/cpu/install-size + idle-GC countdown and Start/Stop/Reset/Remove.
 * Polls while mounted (the workspace uses the container backend).
 */

import { useCallback, useState } from 'react';
import { Box, Loader2, Play, RotateCcw, Square, Trash2 } from 'lucide-react';
import { api, type ContainerAction, type ContainerStatus } from '../../lib/api';
import { useApi } from '../../lib/hooks';
import { useNotify } from '../../contexts/NotificationContext';
import { useConfirm } from '../ui/ConfirmDialog';

const STATE_STYLE: Record<string, string> = {
  running: 'text-success',
  provisioning: 'text-warning',
  stopped: 'text-fg-muted',
  none: 'text-fg-muted',
  unavailable: 'text-error',
};

function idleCountdown(ts?: number | null): string | null {
  if (!ts) return null;
  const days = Math.max(0, (ts * 1000 - Date.now()) / 86400000);
  return days < 1 ? 'soon' : `${Math.round(days)}d`;
}

export function WorkspaceContainerCard({ workspaceId }: { workspaceId: string }) {
  const notify = useNotify();
  const confirm = useConfirm();
  const [busy, setBusy] = useState<ContainerAction | null>(null);
  const { data, refresh } = useApi(
    () => api.getWorkspaceContainer(workspaceId),
    [workspaceId],
    { pollInterval: 5000 },
  );
  const c: ContainerStatus | undefined = data?.container;

  const act = useCallback(async (action: ContainerAction) => {
    if (action === 'reset' || action === 'remove') {
      const ok = await confirm({
        title: action === 'reset' ? 'Reset container?' : 'Remove container?',
        body: action === 'reset'
          ? 'Drops installed packages (workspace files are kept).'
          : 'Deletes the container and its installed state. Files in the workspace are kept.',
        confirmLabel: action === 'reset' ? 'Reset' : 'Remove', danger: true,
      });
      if (!ok) return;
    }
    setBusy(action);
    try {
      await api.workspaceContainerAction(workspaceId, action);
      await refresh();
    } catch (err) {
      notify.notifyError(err, `Container ${action} failed`);
    } finally {
      setBusy(null);
    }
  }, [workspaceId, confirm, notify, refresh]);

  if (c?.state === 'unavailable') {
    return (
      <div className="mx-3 mb-3 rounded-md border border-line bg-surface-sunken p-2.5 text-xs text-fg-muted">
        Container backend selected, but no Docker daemon is reachable (enable <code>shell.docker</code> +
        the dind sidecar, or use it in dev).
      </div>
    );
  }

  const state = c?.state ?? 'none';
  const idle = idleCountdown(c?.idle_gc_at);

  return (
    <div className="mx-3 mb-3 rounded-md border border-line bg-surface-raised p-2.5 text-xs">
      <div className="flex items-center justify-between">
        <span className="flex items-center gap-1.5 font-medium text-fg">
          <Box size={13} /> Container
          <span className={STATE_STYLE[state] ?? 'text-fg-muted'}>· {state}</span>
        </span>
        <div className="flex items-center gap-1">
          {state === 'running'
            ? <IconBtn label="Stop" busy={busy === 'stop'} onClick={() => act('stop')}><Square size={13} /></IconBtn>
            : <IconBtn label="Start" busy={busy === 'start'} onClick={() => act('start')}><Play size={13} /></IconBtn>}
          <IconBtn label="Reset" busy={busy === 'reset'} onClick={() => act('reset')}><RotateCcw size={13} /></IconBtn>
          <IconBtn label="Remove" busy={busy === 'remove'} onClick={() => act('remove')}><Trash2 size={13} /></IconBtn>
        </div>
      </div>
      {state === 'running' && (
        <div className="mt-1.5 flex flex-wrap gap-x-4 gap-y-0.5 text-fg-muted">
          {c?.memory_usage && <span>mem {c.memory_usage}</span>}
          {c?.cpu_percent && <span>cpu {c.cpu_percent}</span>}
          {c?.install_size && <span>size {c.install_size}</span>}
          {idle && <span>idle-GC in {idle}</span>}
        </div>
      )}
      {c?.image && <div className="mt-1 text-[11px] text-fg-muted">{c.image}</div>}
    </div>
  );
}

function IconBtn(
  { label, busy, onClick, children }:
  { label: string; busy: boolean; onClick: () => void; children: React.ReactNode },
) {
  return (
    <button
      type="button" aria-label={label} title={label} onClick={onClick} disabled={busy}
      className="rounded p-1 text-fg-muted hover:bg-surface-hover hover:text-fg disabled:opacity-50"
    >
      {busy ? <Loader2 size={13} className="animate-spin" /> : children}
    </button>
  );
}
