/**
 * useDismissGuard — keep an unsaved edit from being closed away without asking.
 *
 * An editor reports whether it holds unsaved work. While it does, every way of
 * dismissing the stack entry it renders in asks "Discard unsaved changes?"
 * first, and "Keep editing" leaves the entry open:
 *
 * - the shell's close button, Escape and backdrop (they call `closeModal`),
 * - the surface's own close button (it calls the `onClose` ModalPortal gives it),
 * - `dismissTabSurfaces`: ⌘T, the palette's navigation, the TopBar's page pills.
 *
 * The guard registers on the entry (ModalContext's `registerDismissGuard`), so a
 * close path never needs to know which editors live inside the surface. The
 * returned function asks the same question for a close the editor owns itself,
 * such as a dialog nested inside the entry or a detail pane's own close:
 *
 *   const mayDiscard = useDismissGuard(dirty, 'Your edits to notes.md have not been saved.');
 *   const close = async () => { if (await mayDiscard()) onClose(); };
 *
 * A clean editor answers `true` synchronously: it closes at once, with no prompt.
 * Outside a stack entry the hook only returns the function.
 */

import { useCallback, useContext, useEffect, useRef } from 'react';
import { useConfirm } from '../components/ui/ConfirmDialog';
import { ModalContext, ModalEntryContext } from '../contexts/ModalContext';

export function useDismissGuard(dirty: boolean, body: string): () => boolean | Promise<boolean> {
  const confirm = useConfirm();
  const entryId = useContext(ModalEntryContext);
  const register = useContext(ModalContext)?.registerDismissGuard;

  const dirtyRef = useRef(dirty);
  const bodyRef = useRef(body);
  useEffect(() => {
    dirtyRef.current = dirty;
    bodyRef.current = body;
  }, [dirty, body]);

  const mayDismiss = useCallback((): boolean | Promise<boolean> => {
    if (!dirtyRef.current) return true;
    return confirm({
      title: 'Discard unsaved changes?',
      body: bodyRef.current,
      confirmLabel: 'Discard',
      cancelLabel: 'Keep editing',
      danger: true,
    });
  }, [confirm]);

  useEffect(() => {
    if (!entryId || !register) return;
    return register(entryId, mayDismiss);
  }, [entryId, register, mayDismiss]);

  return mayDismiss;
}
