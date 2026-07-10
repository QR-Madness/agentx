import { describe, it, expect, beforeEach } from 'vitest';
import { setActiveServerId, getConversationTabs, type ConversationTab } from './storage';
import {
  RESEARCH_MODE,
  availableThinkingModes,
  DEFAULT_MODE_GATES,
  thinkingModeOf,
  thinkingModeTabPatch,
  thinkingModeWireFields,
} from './thinkingModes';

const baseTab = {
  id: 'tab_1',
  title: 'T',
  sessionId: null,
  profileId: null,
  workflowId: null,
  messages: [],
  isStreaming: false,
  createdAt: 'now',
  lastMessageAt: 'now',
};

function seedTabs(tabs: object[]) {
  localStorage.setItem('agentx:server:srv1:tabs', JSON.stringify(tabs));
}

describe('thinkingModes', () => {
  beforeEach(() => {
    localStorage.clear();
    setActiveServerId('srv1');
  });

  it('backfills thinkingMode from legacy researchMode on load', () => {
    seedTabs([{ ...baseTab, researchMode: true }]);
    expect(getConversationTabs()[0].thinkingMode).toBe(RESEARCH_MODE);
  });

  it('backfills thinkingMode from a legacy thinkingPattern on load', () => {
    seedTabs([{ ...baseTab, thinkingPattern: 'cot' }]);
    expect(getConversationTabs()[0].thinkingMode).toBe('cot');
  });

  it('backfills Auto when neither legacy field is set', () => {
    seedTabs([{ ...baseTab }]);
    expect(getConversationTabs()[0].thinkingMode).toBe('');
  });

  it('a persisted thinkingMode wins over the legacy pair', () => {
    seedTabs([{ ...baseTab, thinkingMode: 'reflection', researchMode: true }]);
    expect(getConversationTabs()[0].thinkingMode).toBe('reflection');
  });

  it('thinkingModeOf prefers thinkingMode, falls back research > pattern > auto', () => {
    expect(thinkingModeOf({ thinkingMode: 'cot', researchMode: true, thinkingPattern: null })).toBe('cot');
    expect(thinkingModeOf({ researchMode: true, thinkingPattern: 'cot' })).toBe(RESEARCH_MODE);
    expect(thinkingModeOf({ thinkingPattern: 'step_back' })).toBe('step_back');
    expect(thinkingModeOf({})).toBe('');
    expect(thinkingModeOf(null)).toBe('');
  });

  it('thinkingModeTabPatch keeps the legacy fields in lockstep', () => {
    expect(thinkingModeTabPatch(RESEARCH_MODE)).toEqual({
      thinkingMode: RESEARCH_MODE, researchMode: true, thinkingPattern: null,
    } satisfies Partial<ConversationTab>);
    expect(thinkingModeTabPatch('cot')).toEqual({
      thinkingMode: 'cot', researchMode: false, thinkingPattern: 'cot',
    });
    expect(thinkingModeTabPatch('')).toEqual({
      thinkingMode: '', researchMode: false, thinkingPattern: null,
    });
  });

  it('derives the unchanged wire contract from the mode', () => {
    expect(thinkingModeWireFields(RESEARCH_MODE)).toEqual({ research_mode: true });
    expect(thinkingModeWireFields('self_consistency')).toEqual({ thinking_pattern: 'self_consistency' });
    expect(thinkingModeWireFields('')).toEqual({});
  });

  it('gates options: patterns kill-switch leaves Auto/Native/Research', () => {
    const opts = availableThinkingModes({ ...DEFAULT_MODE_GATES, patternsEnabled: false });
    expect(opts.map(o => o.value)).toEqual(['', 'native', RESEARCH_MODE]);
  });

  it('gates options: research off drops the Research entry, per-pattern flags drop theirs', () => {
    const opts = availableThinkingModes({
      ...DEFAULT_MODE_GATES, research: false, selfConsistency: false, reflection: false,
    });
    const values = opts.map(o => o.value);
    expect(values).not.toContain(RESEARCH_MODE);
    expect(values).not.toContain('self_consistency');
    expect(values).not.toContain('reflection');
    expect(values).not.toContain('deep_reflection');
    expect(values).toContain('cot');
  });
});
