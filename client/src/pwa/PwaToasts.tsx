/**
 * PwaToasts — bridges the boot-time PWA events (dispatched from registerPwa /
 * installPrompt, which run outside React) into the app's toast system.
 *
 * Rendered once inside the NotificationProvider. Inert under Tauri, where the
 * source events never fire.
 */

import { useEffect } from 'react';
import { useNotify } from '../contexts/NotificationContext';
import { PWA_NEED_REFRESH, PWA_OFFLINE_READY, applyPwaUpdate } from './registerPwa';
import { PWA_INSTALLABLE, PWA_IOS_HINT, promptInstall } from './installPrompt';

export function PwaToasts() {
  const { notify, notifySuccess } = useNotify();

  useEffect(() => {
    const onNeedRefresh = () =>
      notify({
        kind: 'info',
        title: 'Update available',
        message: 'A new version of AgentX is ready.',
        duration: 0, // sticky until the user acts
        action: { label: 'Reload', onClick: applyPwaUpdate },
      });

    const onOfflineReady = () => notifySuccess('AgentX is ready to work offline.');

    const onInstallable = () =>
      notify({
        kind: 'info',
        title: 'Install AgentX',
        message: 'Add AgentX to your device for quick, full-screen access.',
        duration: 15_000,
        action: { label: 'Install', onClick: () => void promptInstall() },
      });

    const onIosHint = () =>
      notify({
        kind: 'info',
        title: 'Install AgentX',
        message: 'Tap the Share icon, then "Add to Home Screen".',
        duration: 12_000,
      });

    window.addEventListener(PWA_NEED_REFRESH, onNeedRefresh);
    window.addEventListener(PWA_OFFLINE_READY, onOfflineReady);
    window.addEventListener(PWA_INSTALLABLE, onInstallable);
    window.addEventListener(PWA_IOS_HINT, onIosHint);
    return () => {
      window.removeEventListener(PWA_NEED_REFRESH, onNeedRefresh);
      window.removeEventListener(PWA_OFFLINE_READY, onOfflineReady);
      window.removeEventListener(PWA_INSTALLABLE, onInstallable);
      window.removeEventListener(PWA_IOS_HINT, onIosHint);
    };
  }, [notify, notifySuccess]);

  return null;
}
