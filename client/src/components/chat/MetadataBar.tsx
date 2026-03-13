/**
 * MetadataBar — Displays message metadata (model, tokens, latency)
 */

import { Zap, Clock, Cpu, User } from 'lucide-react';
import './MetadataBar.css';

export interface MetadataBarProps {
  model?: string;
  tokensInput?: number;
  tokensOutput?: number;
  tokensUsed?: number;  // Legacy single token count
  latencyMs?: number;
  agentName?: string;
}

export function MetadataBar({
  model,
  tokensInput,
  tokensOutput,
  tokensUsed,
  latencyMs,
  agentName,
}: MetadataBarProps) {
  // Don't render if no metadata
  if (!model && !tokensInput && !tokensOutput && !tokensUsed && !latencyMs && !agentName) {
    return null;
  }

  const formatLatency = (ms: number): string => {
    if (ms < 1000) return `${ms}ms`;
    return `${(ms / 1000).toFixed(2)}s`;
  };

  const formatTokens = (): string | null => {
    if (tokensInput !== undefined && tokensOutput !== undefined) {
      return `${tokensInput} in / ${tokensOutput} out`;
    }
    if (tokensUsed !== undefined) {
      return `${tokensUsed} tokens`;
    }
    return null;
  };

  const getModelShortName = (modelName: string): string => {
    // Extract just the model name from paths like "anthropic/claude-3-sonnet"
    const parts = modelName.split('/');
    return parts[parts.length - 1];
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

      {latencyMs !== undefined && (
        <span className="metadata-item latency">
          <Clock size={10} />
          <span>{formatLatency(latencyMs)}</span>
        </span>
      )}
    </div>
  );
}
