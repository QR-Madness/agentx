import { describe, it, expect } from 'vitest';
import { voicesFor } from './voiceCatalog';

describe('voicesFor', () => {
  it('returns curated voices for OpenAI TTS models', () => {
    const voices = voicesFor('openai/gpt-4o-mini-tts');
    expect(voices.length).toBeGreaterThan(0);
    expect(voices.map((v) => v.id)).toContain('nova');
  });

  it('matches regardless of a provider prefix or casing', () => {
    expect(voicesFor('OpenRouter:Microsoft/MAI-Voice-2').map((v) => v.id)).toContain(
      'en-US-Harper:MAI-Voice-2',
    );
    expect(voicesFor('x-ai/grok-voice-tts-1.0').map((v) => v.id)).toContain('Eve');
  });

  it('returns [] for unknown models and nullish input', () => {
    expect(voicesFor('some/unknown-model')).toEqual([]);
    expect(voicesFor(null)).toEqual([]);
    expect(voicesFor(undefined)).toEqual([]);
  });
});
