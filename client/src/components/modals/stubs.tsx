/**
 * Modal content components — wrappers for tabs rendered in modals/drawers
 */

import { SettingsTab } from '../tabs/SettingsTab';
import { MemoryTab } from '../tabs/MemoryTab';
import { ToolsTab } from '../tabs/ToolsTab';
import { TranslationTab } from '../tabs/TranslationTab';

interface ModalContentProps {
  onClose: () => void;
}

export function SettingsModalContent({ onClose: _onClose }: ModalContentProps) {
  return (
    <div className="modal-content-wrapper">
      <SettingsTab />
    </div>
  );
}

export function MemoryModalContent({ onClose: _onClose }: ModalContentProps) {
  return (
    <div className="modal-content-wrapper">
      <MemoryTab />
    </div>
  );
}

export function ToolsModalContent({ onClose: _onClose }: ModalContentProps) {
  return (
    <div className="modal-content-wrapper">
      <ToolsTab />
    </div>
  );
}

export function TranslationModalContent({ onClose: _onClose }: ModalContentProps) {
  return (
    <div className="modal-content-wrapper">
      <TranslationTab />
    </div>
  );
}

/**
 * Fallback stub for unregistered components
 */
export function StubModal({ onClose }: ModalContentProps) {
  return (
    <div style={{ padding: 24 }}>
      <h3 style={{ color: 'var(--text-primary)', marginBottom: 12 }}>Coming Soon</h3>
      <p style={{ color: 'var(--text-secondary)', marginBottom: 16 }}>
        This panel will be implemented in a future phase.
      </p>
      <button className="button-secondary" onClick={onClose}>
        Close
      </button>
    </div>
  );
}
