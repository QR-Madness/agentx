/**
 * ConfirmDialog + useConfirm — a themed, promise-based replacement for the
 * clanky native `window.confirm`.
 *
 *   const confirm = useConfirm();
 *   if (await confirm({ title: 'Delete?', body: '…', danger: true })) { … }
 *
 * A single dialog is mounted by `ConfirmProvider` at the app root, so it survives
 * the caller's own UI closing (e.g. a row dropdown menu) and renders above
 * everything via the Dialog portal. Enter confirms, Esc / backdrop cancels.
 */

import { createContext, useCallback, useContext, useRef, useState, type ReactNode } from 'react';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogFooter,
  DialogTitle,
  DialogDescription,
} from './Dialog';
import { Button } from './Button';

export interface ConfirmOptions {
  title: string;
  body?: ReactNode;
  confirmLabel?: string;
  cancelLabel?: string;
  danger?: boolean;
}

type ConfirmFn = (opts: ConfirmOptions) => Promise<boolean>;

const ConfirmContext = createContext<ConfirmFn | null>(null);

export function ConfirmProvider({ children }: { children: ReactNode }) {
  const [open, setOpen] = useState(false);
  const [opts, setOpts] = useState<ConfirmOptions | null>(null);
  const resolver = useRef<((v: boolean) => void) | null>(null);

  const settle = useCallback((value: boolean) => {
    resolver.current?.(value);
    resolver.current = null;
    setOpen(false);
  }, []);

  const confirm = useCallback<ConfirmFn>((next) => {
    setOpts(next);
    setOpen(true);
    return new Promise<boolean>((resolve) => {
      resolver.current = resolve;
    });
  }, []);

  return (
    <ConfirmContext.Provider value={confirm}>
      {children}
      <Dialog open={open} onOpenChange={(o) => { if (!o) settle(false); }}>
        <DialogContent showClose={false} className="max-w-md">
          <DialogHeader>
            <DialogTitle>{opts?.title}</DialogTitle>
            {opts?.body && <DialogDescription>{opts.body}</DialogDescription>}
          </DialogHeader>
          <DialogFooter>
            <Button variant="ghost" onClick={() => settle(false)}>
              {opts?.cancelLabel ?? 'Cancel'}
            </Button>
            <Button
              variant={opts?.danger ? 'danger' : 'primary'}
              autoFocus
              onClick={() => settle(true)}
            >
              {opts?.confirmLabel ?? 'Confirm'}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </ConfirmContext.Provider>
  );
}

export function useConfirm(): ConfirmFn {
  const ctx = useContext(ConfirmContext);
  if (!ctx) throw new Error('useConfirm must be used within a ConfirmProvider');
  return ctx;
}
