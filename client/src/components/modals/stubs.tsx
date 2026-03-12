/**
 * Stub modal component — placeholder until real modals are built in Phase 13.5
 */

export function StubModal({ onClose }: { onClose: () => void }) {
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
