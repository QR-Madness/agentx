/**
 * useOpenAmbassador — open the Ambassador the right way for the viewport.
 *
 * On a wide screen the Ambassador docks as a resizable side column next to the
 * conversation (parallel, no dim/blur — see `AmbassadorDock`). When there's no
 * room to dock (narrow screens / mobile) it falls back to the existing
 * full-screen sheet via the modal system. One entry point, two presentations, so
 * the per-message button and the command palette can't drift.
 */

import { useCallback } from 'react';
import { useModal } from '../contexts/ModalContext';
import { useAmbassadorDock } from '../contexts/AmbassadorDockContext';
import { SURFACES } from '../lib/surfaces';

export function useOpenAmbassador(): () => void {
  const { openModal } = useModal();
  const { dockCapable, setOpen } = useAmbassadorDock();
  return useCallback(() => {
    if (dockCapable) setOpen(true);
    else openModal(SURFACES.ambassador);
  }, [dockCapable, setOpen, openModal]);
}
