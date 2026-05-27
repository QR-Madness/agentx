import { useState, useEffect } from 'react';
import {
  Layers,
  Server,
  RefreshCw,
  AlertTriangle,
  Save,
} from 'lucide-react';
import { api } from '../../../lib/api';
import { useNotify } from '../../../contexts/NotificationContext';
import { Button, Card, Badge, SectionHeader, Input } from '../../ui';

interface ContextLimits {
  lmstudio: { context_window: number; max_output_tokens: number };
  models: Record<string, { context_window: number; max_output_tokens: number }>;
}

export default function ModelsSection() {
  const { notifyError, notifySuccess } = useNotify();
  const [contextLimits, setContextLimits] = useState<ContextLimits | null>(null);
  const [loadingContextLimits, setLoadingContextLimits] = useState(false);
  const [savingContextLimits, setSavingContextLimits] = useState(false);

  useEffect(() => {
    fetchContextLimits();
  }, []);

  const fetchContextLimits = async () => {
    setLoadingContextLimits(true);
    try {
      const limits = await api.getContextLimits();
      setContextLimits(limits);
    } catch (error) {
      notifyError(error, 'Failed to load context limits');
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
    try {
      await api.updateContextLimits(contextLimits);
      notifySuccess('Context limits saved', 'Models');
    } catch (error) {
      notifyError(error, 'Failed to save context limits');
    } finally {
      setSavingContextLimits(false);
    }
  };

  return (
    <div className="settings-section fade-in">
      <SectionHeader
        icon={<Layers size={20} />}
        title="Model Context Limits"
        description="Configure context window and output token limits for local models (LM Studio)"
        actions={
          <Button
            variant="primary"
            onClick={handleSaveContextLimits}
            loading={savingContextLimits}
            disabled={savingContextLimits || !contextLimits}
          >
            <Save size={16} />
            Save Limits
          </Button>
        }
      />

      {loadingContextLimits ? (
        <Card className="empty-state">
          <RefreshCw size={24} className="spin" />
          <p>Loading context limits...</p>
        </Card>
      ) : contextLimits ? (
        <div className="providers-list">
          <Card className="provider-card">
            <div className="provider-header">
              <div className="provider-info">
                <div className="provider-icon local">
                  <Server size={20} />
                </div>
                <div>
                  <h3 className="provider-name">
                    LM Studio
                    <Badge variant="neutral" size="sm">Local</Badge>
                  </h3>
                  <p className="provider-description">
                    Hardware limits for local models (API providers use their own per-model capabilities)
                  </p>
                </div>
              </div>
            </div>
            <div className="context-limits-form">
              <div className="form-group">
                <label htmlFor="ctx-window">Context Window (tokens)</label>
                <Input
                  id="ctx-window"
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
                <label htmlFor="ctx-output">Max Output Tokens</label>
                <Input
                  id="ctx-output"
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
          </Card>
        </div>
      ) : (
        <Card className="empty-state">
          <AlertTriangle size={32} />
          <p>Failed to load context limits</p>
          <Button variant="secondary" onClick={fetchContextLimits}>
            <RefreshCw size={16} />
            Retry
          </Button>
        </Card>
      )}
    </div>
  );
}
