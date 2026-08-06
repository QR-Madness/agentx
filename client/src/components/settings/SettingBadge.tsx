/**
 * SettingBadge — one vocabulary for the labels beside a setting.
 *
 * These grew ad hoc and stopped meaning anything consistent: "Recommended" was
 * green, "LLM Required" amber, and "Experimental" was passed with no variant at
 * all, so the riskiest label rendered as the quietest one. The kinds below are
 * the whole vocabulary; each maps to exactly one colour everywhere.
 */

import { Badge, type BadgeProps } from '../ui';

export type SettingBadgeKind =
  /** On by default and worth keeping — the safe choice. */
  | 'recommended'
  /** Costs a model call when it runs. */
  | 'llm-required'
  /** Unproven; may change or be withdrawn. */
  | 'experimental'
  /** Works, still being shaped. */
  | 'beta'
  /** Runs on this machine, not a hosted provider. */
  | 'local';

const BADGES: Record<SettingBadgeKind, { text: string; variant: BadgeProps['variant'] }> = {
  recommended: { text: 'Recommended', variant: 'success' },
  'llm-required': { text: 'LLM Required', variant: 'warning' },
  experimental: { text: 'Experimental', variant: 'danger' },
  beta: { text: 'Beta', variant: 'accent' },
  local: { text: 'Local', variant: 'neutral' },
};

export function SettingBadge({
  kind,
  size = 'sm',
}: {
  kind: SettingBadgeKind;
  size?: BadgeProps['size'];
}) {
  const { text, variant } = BADGES[kind];
  return (
    <Badge variant={variant} size={size}>
      {text}
    </Badge>
  );
}

/** The badge a manifest tier implies, if any. `essential` gets none. */
export function tierBadge(tier?: string): SettingBadgeKind | null {
  if (tier === 'experimental') return 'experimental';
  return null;
}
