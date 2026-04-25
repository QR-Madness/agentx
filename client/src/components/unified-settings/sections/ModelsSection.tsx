import { useState, useEffect } from 'react';
import {
  Layers,
  Server,
  RefreshCw,
  Check,
  AlertTriangle,
  Save,
} from 'lucide-react';
import { api } from '../../../lib/api';

interface ContextLimits {
  lmstudio: { context_window: number; max_output_tokens: number };
  models: Record<string, { context_window: number; max_output_tokens: number }>;
}

export default function ModelsSection() {
  const [contextLimits, setContextLimits] = useState<ContextLimits | null>(null);
  const [loadingContextLimits, setLoadingContextLimits] = useState(false);
  const [savingContextLimits, setSavingContextLimits] = useState(false);
  const [contextLimitsMessage, setContextLimitsMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null);

  useEffect(() => {
    fetchContextLimits();
  }, []);

  const fetchContextLimits = async () => {
    setLoadingContextLimits(true);
    try {
      const limits = await api.getContextLimits();
      setContextLimits(limits);
    } catch (error) {
      console.error('Failed to fetch context limits:', error);
    } finally {
      setLoadingContextLimits(false);
    }
  };

  const handleContextLimitChange = (field: 'context_window' | 'max_output_tokens', value: number) => {
    if (!contextLimits) return;
    setContextLimits({
      ...contextLimits,
      lmstudio: { ...contextLimits.lmstudio, [field]: value },
    });
  };

  const handleSaveContextLimits = async () => {
    if (!contextLimits) return;

    setSavingContextLimits(true);
    setContextLimitsMessage(null);

    try {
      await api.updateContextLimits(contextLimits);
      setContextLimitsMessage({ type: 'success', text: 'Context limits saved' });
      setTimeout(() => setContextLimitsMessage(null), 3000);
    } catch (error) {
      console.error('Failed to save context limits:', error);
      setContextLimitsMessage({ type: 'error', text: 'Failed to save context limits' });
    } finally {
      setSavingContextLimits(false);
    }
  };

  return (
    <div className="settings-section fade-in">
      <div className="section-header">
        <div>
          <h2 className="section-title">
            <Layers size={20} className="section-title-icon" />
            Model Context Limits
          </h2>
          <p className="section-description">
            Configure context window and output token limits for local models (LM Studio)
          </p>
        </div>
        <button
          className="button-primary"
          onClick={handleSaveContextLimits}
          disabled={savingContextLimits || !contextLimits}
        >
          {savingContextLimits ? (
            <RefreshCw size={16} className="spin" />
          ) : (
            <Save size={16} />
          )}
          Save Limits
        </button>
      </div>

      {contextLimitsMessage && (
        <div className={`config-message ${contextLimitsMessage.type}`}>
          {contextLimitsMessage.type === 'success' ? <Check size={16} /> : <AlertTriangle size={16} />}
          <span>{contextLimitsMessage.text}</span>
        </div>
      )}

      {loadingContextLimits ? (
        <div className="card" style={{ padding: '2rem', textAlign: 'center' }}>
          <RefreshCw size={24} className="spin" style={{ marginBottom: '0.5rem' }} />
          <p>Loading context limits...</p>
        </div>
      ) : contextLimits ? (
        <div className="providers-list">
          <div className="provider-card card">
            <div className="provider-header">
              <div className="provider-info">
                <div className="provider-icon local">
                  <Server size={20} />
                </div>
                <div>
                  <h3 className="provider-name">
                    LM Studio
                    <span className="provider-badge local">Local</span>
                  </h3>
                  <p className="provider-description">
                    Hardware limits for local models (API providers use their own per-model capabilities)
                  </p>
                </div>
              </div>
            </div>
            <div className="context-limits-form">
              <div className="form-group">
                <label>Context Window (tokens)</label>
                <input
                  type="number"
                  value={contextLimits.lmstudio.context_window}
                  onChange={(e) => handleContextLimitChange('context_window', parseInt(e.target.value) || 0)}
                  min={1024}
                  max={1000000}
                  step={1024}
                />
                <span className="form-hint">
                  {(contextLimits.lmstudio.context_window / 1000).toFixed(0)}k tokens
                </span>
              </div>
              <div className="form-group">
                <label>Max Output Tokens</label>
                <input
                  type="number"
                  value={contextLimits.lmstudio.max_output_tokens}
                  onChange={(e) => handleContextLimitChange('max_output_tokens', parseInt(e.target.value) || 0)}
                  min={256}
                  max={131072}
                  step={256}
                />
                <span className="form-hint">
                  {(contextLimits.lmstudio.max_output_tokens / 1000).toFixed(0)}k tokens
                </span>
              </div>
            </div>
          </div>
        </div>
      ) : (
        <div className="empty-state card">
          <AlertTriangle size={32} />
          <p>Failed to load context limits</p>
          <button className="button-secondary" onClick={fetchContextLimits}>
            <RefreshCw size={16} />
            Retry
          </button>
        </div>
      )}
    </div>
  );
}
