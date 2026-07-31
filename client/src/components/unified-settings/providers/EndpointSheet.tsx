/**
 * EndpointSheet — point AgentX at any OpenAI-compatible endpoint.
 *
 * The presets are a convenience, not an integration: each one only pre-fills an
 * id, a label and a base URL. Nothing about a preset is special-cased in the
 * backend — "Custom" reaches exactly the same code path, which is the point of
 * the provider catalog.
 *
 * Test-before-save is deliberate. A wrong base URL is otherwise invisible until
 * a turn fails much later, so the sheet offers to reach the endpoint and count
 * its models while the user still has the form open.
 */

import { useState } from 'react';
import { Check, Plus, Wifi, X } from 'lucide-react';
import { api, apiErrorMessage } from '../../../lib/api';
import type { ProviderTestResult } from '../../../lib/api';
import { useNotify } from '../../../contexts/NotificationContext';
import { Button, Dialog, DialogContent, DialogHeader, DialogTitle, Input } from '../../ui';

interface Preset {
  id: string;
  label: string;
  base_url: string;
  /** Shown under the picker once selected — where to find the key, mostly. */
  hint?: string;
}

/** Data only: id + label + base URL. Add a row, get a preset. */
const PRESETS: Preset[] = [
  { id: 'groq', label: 'Groq', base_url: 'https://api.groq.com/openai/v1' },
  { id: 'together', label: 'Together', base_url: 'https://api.together.xyz/v1' },
  { id: 'deepseek', label: 'DeepSeek', base_url: 'https://api.deepseek.com/v1' },
  { id: 'fireworks', label: 'Fireworks', base_url: 'https://api.fireworks.ai/inference/v1' },
  { id: 'xai', label: 'xAI', base_url: 'https://api.x.ai/v1' },
  { id: 'mistral', label: 'Mistral', base_url: 'https://api.mistral.ai/v1' },
  { id: 'cerebras', label: 'Cerebras', base_url: 'https://api.cerebras.ai/v1' },
  {
    id: 'ollama',
    label: 'Ollama',
    base_url: 'http://localhost:11434/v1',
    hint: 'Runs on your machine — no key needed.',
  },
  {
    id: 'vllm',
    label: 'vLLM',
    base_url: 'http://localhost:8000/v1',
    hint: 'Runs on your machine — no key needed.',
  },
  { id: '', label: 'Custom', base_url: '' },
];

export interface EndpointSheetProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  /** Ids already registered (built-in or custom) — rejected client-side too. */
  takenIds: string[];
  onSaved: () => void;
}

export function EndpointSheet({ open, onOpenChange, takenIds, onSaved }: EndpointSheetProps) {
  const { notifyError, notifySuccess } = useNotify();
  const [preset, setPreset] = useState<Preset>(PRESETS[PRESETS.length - 1]);
  const [id, setId] = useState('');
  const [label, setLabel] = useState('');
  const [baseUrl, setBaseUrl] = useState('');
  const [apiKey, setApiKey] = useState('');
  const [testing, setTesting] = useState(false);
  const [saving, setSaving] = useState(false);
  const [result, setResult] = useState<ProviderTestResult | null>(null);

  const reset = () => {
    setPreset(PRESETS[PRESETS.length - 1]);
    setId('');
    setLabel('');
    setBaseUrl('');
    setApiKey('');
    setResult(null);
  };

  const choosePreset = (next: Preset) => {
    setPreset(next);
    setId(next.id);
    setLabel(next.label === 'Custom' ? '' : next.label);
    setBaseUrl(next.base_url);
    setResult(null);
  };

  const idTaken = takenIds.includes(id.trim().toLowerCase());
  const idMalformed = id.trim().length > 0 && !/^[a-z0-9][a-z0-9_-]{1,31}$/.test(id.trim().toLowerCase());
  const canSubmit = id.trim().length > 1 && baseUrl.trim().length > 0 && !idTaken && !idMalformed;

  const handleTest = async () => {
    setTesting(true);
    setResult(null);
    try {
      setResult(await api.testProvider({ base_url: baseUrl.trim(), api_key: apiKey || undefined, id: id.trim() }));
    } catch (error) {
      notifyError(error, "Couldn't test the endpoint");
    } finally {
      setTesting(false);
    }
  };

  const handleSave = async () => {
    setSaving(true);
    try {
      await api.saveCustomProvider({
        id: id.trim().toLowerCase(),
        label: label.trim() || undefined,
        base_url: baseUrl.trim(),
        api_key: apiKey || undefined,
      });
      notifySuccess(`${label.trim() || id.trim()} connected`, 'Providers');
      reset();
      onOpenChange(false);
      onSaved();
    } catch (error) {
      notifyError(apiErrorMessage(error), "Couldn't save the endpoint");
    } finally {
      setSaving(false);
    }
  };

  return (
    <Dialog
      open={open}
      onOpenChange={(next) => {
        if (!next) reset();
        onOpenChange(next);
      }}
    >
      <DialogContent className="endpoint-sheet">
        <DialogHeader>
          <DialogTitle>Connect an endpoint</DialogTitle>
        </DialogHeader>

        <p className="endpoint-sheet-lede">
          Point AgentX at any OpenAI-compatible endpoint. Pick a preset to fill in the address, or
          enter your own.
        </p>

        <div className="endpoint-presets" role="group" aria-label="Endpoint presets">
          {PRESETS.map((option) => (
            <button
              key={option.label}
              type="button"
              className={`endpoint-preset${preset.label === option.label ? ' is-selected' : ''}`}
              onClick={() => choosePreset(option)}
            >
              {option.label}
            </button>
          ))}
        </div>
        {preset.hint && <p className="provider-note">{preset.hint}</p>}

        <div className="endpoint-fields">
          <label className="connection-field-label" htmlFor="endpoint-id">
            Name in model references
          </label>
          <Input
            id="endpoint-id"
            value={id}
            onChange={(event) => setId(event.target.value)}
            placeholder="groq"
            autoComplete="off"
            spellCheck={false}
          />
          <p className="endpoint-help">
            {idTaken ? (
              <span className="text-error">
                &lsquo;{id.trim().toLowerCase()}&rsquo; is already in use — pick another name.
              </span>
            ) : idMalformed ? (
              <span className="text-error">
                Use 2–32 characters: lowercase letters, digits, &lsquo;-&rsquo; or &lsquo;_&rsquo;.
              </span>
            ) : (
              <>
                Used in model names, like <code>{(id.trim() || 'groq')}:llama-3.3-70b</code>.
              </>
            )}
          </p>

          <label className="connection-field-label" htmlFor="endpoint-label">
            Display name <span className="endpoint-optional">optional</span>
          </label>
          <Input
            id="endpoint-label"
            value={label}
            onChange={(event) => setLabel(event.target.value)}
            placeholder="Groq Cloud"
            autoComplete="off"
          />

          <label className="connection-field-label" htmlFor="endpoint-url">
            Base URL
          </label>
          <Input
            id="endpoint-url"
            value={baseUrl}
            onChange={(event) => {
              setBaseUrl(event.target.value);
              setResult(null);
            }}
            placeholder="https://api.groq.com/openai/v1"
            autoComplete="off"
            spellCheck={false}
          />

          <label className="connection-field-label" htmlFor="endpoint-key">
            API key <span className="endpoint-optional">optional for local servers</span>
          </label>
          <Input
            id="endpoint-key"
            type="password"
            value={apiKey}
            onChange={(event) => setApiKey(event.target.value)}
            placeholder="gsk_…"
            autoComplete="off"
            spellCheck={false}
          />
        </div>

        {result && (
          <p className={`connection-test ${result.reachable ? 'ok' : 'bad'}`}>
            {result.reachable ? <Check size={14} aria-hidden /> : <X size={14} aria-hidden />}
            {result.reachable
              ? `Reached in ${result.elapsed_ms} ms — ${result.models_available.toLocaleString()} models.`
              : `Couldn't reach ${result.base_url} — ${result.error ?? 'no response'}.`}
          </p>
        )}

        <div className="connection-actions endpoint-sheet-actions">
          <Button
            variant="secondary"
            onClick={handleTest}
            loading={testing}
            disabled={!baseUrl.trim() || testing}
          >
            <Wifi size={16} />
            {testing ? 'Testing…' : 'Test connection'}
          </Button>
          <Button variant="primary" onClick={handleSave} loading={saving} disabled={!canSubmit || saving}>
            <Plus size={16} />
            {saving ? 'Connecting…' : 'Connect'}
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
}
