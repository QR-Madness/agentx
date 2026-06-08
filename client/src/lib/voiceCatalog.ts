/**
 * voiceCatalog — per-speech-model voice options for the voice picker.
 *
 * OpenRouter has no programmatic voices API; voices are model-specific and shown
 * only on each model's page. Fixed-set TTS models are curated here (matched by a
 * substring of the model id); models with large/open voice sets (e.g. MAI-Voice-2's
 * Azure locale names) ship a small starter set and lean on the picker's free-text
 * "Custom…" escape. Unknown models return `[]` → the picker is pure free-text.
 *
 * If OpenRouter ever exposes voices on the models API, swap `voicesFor` for a
 * fetch without touching callers.
 */

export interface VoiceOption {
  id: string;
  label: string;
}

interface CatalogEntry {
  /** Lowercased substrings of the model id this entry matches. */
  match: string[];
  voices: VoiceOption[];
}

const v = (id: string, label?: string): VoiceOption => ({ id, label: label ?? id });

const CATALOG: CatalogEntry[] = [
  {
    // OpenAI TTS family (tts-1, tts-1-hd, gpt-4o-mini-tts, gpt-4o-audio…)
    match: ['openai/tts', 'openai/gpt-4o-mini-tts', 'openai/gpt-4o-audio', 'openai/gpt-audio'],
    voices: [
      v('alloy'), v('ash'), v('ballad'), v('coral'), v('echo'),
      v('fable'), v('nova'), v('onyx'), v('sage'), v('shimmer'),
    ],
  },
  {
    // xAI Grok Voice TTS
    match: ['grok-voice', 'x-ai/grok-voice'],
    voices: [v('Eve'), v('Ara'), v('Rex'), v('Sal'), v('Leo')],
  },
  {
    // Microsoft MAI-Voice-2 — Azure locale names (en-US-<Name>:MAI-Voice-2);
    // a verified starter set, free-text covers the rest.
    match: ['mai-voice-2', 'microsoft/mai-voice'],
    voices: [
      v('en-US-Harper:MAI-Voice-2', 'Harper (en-US)'),
      v('en-US-Ava:MAI-Voice-2', 'Ava (en-US)'),
      v('en-US-Andrew:MAI-Voice-2', 'Andrew (en-US)'),
      v('en-US-Emma:MAI-Voice-2', 'Emma (en-US)'),
      v('en-US-Brian:MAI-Voice-2', 'Brian (en-US)'),
    ],
  },
  {
    // Canopy Labs Orpheus
    match: ['orpheus'],
    voices: [v('tara'), v('leah'), v('jess'), v('leo'), v('dan'), v('mia'), v('zac')],
  },
  {
    // Kokoro — a representative en subset of its 50+ voices.
    match: ['kokoro'],
    voices: [
      v('af_heart', 'Heart (af)'), v('af_bella', 'Bella (af)'), v('af_nova', 'Nova (af)'),
      v('am_michael', 'Michael (am)'), v('am_puck', 'Puck (am)'),
      v('bf_emma', 'Emma (bf)'), v('bm_george', 'George (bm)'),
    ],
  },
];

/** Known voice options for a `provider:model` (or bare model) id. Empty if unknown. */
export function voicesFor(modelId: string | null | undefined): VoiceOption[] {
  if (!modelId) return [];
  const id = modelId.toLowerCase();
  const entry = CATALOG.find((e) => e.match.some((m) => id.includes(m)));
  return entry ? entry.voices : [];
}
