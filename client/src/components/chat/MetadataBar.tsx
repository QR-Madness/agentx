/**
 * MetadataBar — Displays message metadata (model, tokens, cost, latency)
 */

import { useEffect, useState } from 'react';
import { Zap, Clock, Cpu, User, DollarSign, Brain, Telescope } from 'lucide-react';
import { fetchModelsOnce } from '../common/modelCatalog';
import { formatCost, formatLatency, getModelShortName } from '../../lib/format';
import { THINKING_MODE_LABELS } from '../../lib/thinkingModes';
import './MetadataBar.css';

export interface MetadataBarProps {
  model?: string;
  tokensInput?: number;
  tokensOutput?: number;
  tokensUsed?: number;  // Legacy single token count
  costEstimate?: number;
  costCurrency?: string;
  latencyMs?: number;
  agentName?: string;
  /** Thinking pattern the turn ran with — renders a small mode badge. */
  thinkingPattern?: string;
  /** Turn ran in Research Mode (patterns suppressed) — Research badge wins. */
  research?: boolean;
}

interface DerivedCost {
  amount: number;
  currency: string;
}

export function MetadataBar({
  model,
  tokensInput,
  tokensOutput,
  tokensUsed,
  costEstimate,
  costCurrency,
  latencyMs,
  agentName,
  thinkingPattern,
  research,
}: MetadataBarProps) {
  // Backfill cost when the backend didn't compute one (older turns, or
  // providers that don't populate pricing in get_capabilities — e.g. the
  // built-in Anthropic provider). Looks up the model in the cached
  // /api/providers/models payload and derives cost from tokens.
  const [derivedCost, setDerivedCost] = useState<DerivedCost | null>(null);

  const totalTokens =
    (tokensInput ?? 0) + (tokensOutput ?? 0) || tokensUsed || 0;
  const needsBackfill =
    costEstimate === undefined && !!model && totalTokens > 0;

  useEffect(() => {
    if (!needsBackfill) {
      setDerivedCost(null);
      return;
    }
    let cancelled = false;
    fetchModelsOnce().then((models) => {
      if (cancelled) return;
      const info = models.find((m) => m.id === model);
      const inRate = info?.cost_per_1k_input ?? null;
      const outRate = info?.cost_per_1k_output ?? null;
      if (inRate == null && outRate == null) return;
      const inTokens = tokensInput ?? totalTokens;
      const outTokens = tokensOutput ?? 0;
      const amount =
        ((inRate ?? 0) * inTokens) / 1000 + ((outRate ?? 0) * outTokens) / 1000;
      if (amount <= 0) return;
      setDerivedCost({
        amount,
        currency: info?.pricing_currency || 'USD',
      });
    });
    return () => {
      cancelled = true;
    };
  }, [needsBackfill, model, tokensInput, tokensOutput, totalTokens]);

  const effectiveCost = costEstimate ?? derivedCost?.amount;
  const effectiveCurrency = costCurrency ?? derivedCost?.currency ?? 'USD';
  const hasCost = effectiveCost !== undefined && effectiveCost > 0;

  // The mode badge: Research wins (patterns are suppressed on research turns);
  // otherwise the pattern the turn actually ran with. 'native' stays unbadged —
  // it's the ordinary case, a badge would be noise.
  const modeBadge = research
    ? { icon: <Telescope size={10} />, label: 'Research' }
    : thinkingPattern && thinkingPattern !== 'native'
      ? { icon: <Brain size={10} />, label: THINKING_MODE_LABELS[thinkingPattern] ?? thinkingPattern }
      : null;

  // Don't render if no metadata
  if (
    !model && !tokensInput && !tokensOutput && !tokensUsed
    && !hasCost && !latencyMs && !agentName && !modeBadge
  ) {
    return null;
  }

  const formatTokens = (): string | null => {
    if (tokensInput !== undefined && tokensOutput !== undefined) {
      return `${tokensInput} in / ${tokensOutput} out`;
    }
    if (tokensUsed !== undefined) {
      return `${tokensUsed} tokens`;
    }
    return null;
  };

  const tokenDisplay = formatTokens();

  return (
    <div className="metadata-bar">
      {agentName && (
        <span className="metadata-item agent">
          <User size={10} />
          <span>{agentName}</span>
        </span>
      )}

      {modeBadge && (
        <span
          className="metadata-item mode"
          title={research
            ? 'This turn ran in Research Mode (thinking patterns don’t apply there)'
            : 'Thinking pattern this turn ran with'}
        >
          {modeBadge.icon}
          <span>{modeBadge.label}</span>
        </span>
      )}

      {model && (
        <span className="metadata-item model">
          <Cpu size={10} />
          <span>{getModelShortName(model)}</span>
        </span>
      )}

      {tokenDisplay && (
        <span className="metadata-item tokens">
          <Zap size={10} />
          <span>{tokenDisplay}</span>
        </span>
      )}

      {hasCost && (
        <span
          className="metadata-item cost"
          title="Estimated cost for this turn, based on the model's listed pricing."
        >
          <DollarSign size={10} />
          <span>~{formatCost(effectiveCost!, effectiveCurrency)}</span>
        </span>
      )}

      {latencyMs !== undefined && (
        <span className="metadata-item latency">
          <Clock size={10} />
          <span>{formatLatency(latencyMs)}</span>
        </span>
      )}
    </div>
  );
}
