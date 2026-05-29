/**
 * GateChrome — the frameless-window titlebar strip for the pre-app "gate"
 * screens (Connect/Auth and VersionMismatch), which render before the main
 * TopBar exists. Without this, those pages have no drag region and no
 * min/max/close on Windows/Linux — the window looks chromeless and can't be
 * moved. Mirrors the chrome TopBar provides for the authenticated app.
 *
 * Renders nothing in the browser build (`!isTauri`), where the OS frame applies.
 */

import { WindowControls } from './WindowControls';
import { isTauri, showWindowControls } from '../lib/platform';
import './GateChrome.css';

export function GateChrome() {
  // Outside Tauri the browser draws its own frame — no custom strip needed.
  if (!isTauri) return null;

  return (
    <div className="gate-chrome" data-tauri-drag-region>
      {showWindowControls && <WindowControls />}
    </div>
  );
}
