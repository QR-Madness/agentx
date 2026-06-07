/**
 * agentAccent — a deterministic, quiet identity color for an agent.
 *
 * Hashes the immutable `agent_id` to a stable hue so every agent has a recognizable
 * signature color that we can thread through the UI (hero aura, nav ring, kind badge —
 * and later chat bubbles / Alloy traces). Saturation/lightness are kept restrained so
 * the accent harmonizes with all three themes, including monochrome Professional, and
 * never fights the chrome.
 */

export interface AgentAccent {
  /** The signature color (CSS hsl). */
  accent: string;
  /** A soft, low-alpha tint of the same hue — for auras/backgrounds. */
  soft: string;
  /** The raw hue (0–359), if a caller wants to derive its own shades. */
  hue: number;
}

/** Stable string hash (djb2) → unsigned 32-bit. */
function hashString(str: string): number {
  let h = 5381;
  for (let i = 0; i < str.length; i++) {
    h = ((h << 5) + h + str.charCodeAt(i)) | 0;
  }
  return h >>> 0;
}

export function agentAccent(agentId: string | undefined | null): AgentAccent {
  const hue = hashString(agentId || 'agent') % 360;
  // Restrained S/L: vivid enough to read on dark + light, calm enough for Professional.
  return {
    hue,
    accent: `hsl(${hue} 52% 60%)`,
    soft: `hsl(${hue} 52% 60% / 0.14)`,
  };
}
