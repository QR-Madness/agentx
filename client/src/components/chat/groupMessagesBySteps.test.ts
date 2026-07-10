import { describe, it, expect } from 'vitest';
import { foldToolRuns, groupMessagesBySteps } from './groupMessagesBySteps';
import type { ConversationMessage, PlanStepRef } from '../../lib/messages';

let seq = 0;
function step(subtaskId: number, planId = 'p1'): PlanStepRef {
  return {
    planId,
    subtaskId,
    subtaskIndex: subtaskId + 1,
    subtaskCount: 3,
    subtaskTitle: `Step ${subtaskId + 1}`,
  };
}

function assistant(planStep?: PlanStepRef): ConversationMessage {
  return {
    id: `m${seq++}`,
    type: 'assistant',
    content: 'x',
    timestamp: '2026-05-28T00:00:00.000Z',
    planStep,
  };
}

function toolCall(planStep?: PlanStepRef): ConversationMessage {
  return {
    id: `t${seq++}`,
    type: 'tool_call',
    toolName: 'web_search',
    toolCallId: `call-${seq}`,
    arguments: {},
    status: 'completed',
    timestamp: '2026-05-28T00:00:00.000Z',
    planStep,
  } as ConversationMessage;
}

describe('foldToolRuns', () => {
  it('stacks runs of ≥2 consecutive tool calls and demotes singles', () => {
    const out = foldToolRuns([
      assistant(),
      toolCall(), toolCall(), toolCall(),
      assistant(),
      toolCall(),
    ]);
    expect(out.map(i => i.kind)).toEqual(['message', 'toolRun', 'message', 'message']);
    const run = out[1];
    if (run.kind === 'toolRun') expect(run.messages).toHaveLength(3);
  });

  it('keys a run by its first call id', () => {
    const a = toolCall();
    const out = foldToolRuns([a, toolCall()]);
    expect(out[0].kind).toBe('toolRun');
    if (out[0].kind === 'toolRun') expect(out[0].key).toBe(`toolrun-${a.id}`);
  });
});

describe('groupMessagesBySteps', () => {
  it('passes through messages without planStep as standalone items', () => {
    const out = groupMessagesBySteps([assistant(), assistant()]);
    expect(out).toHaveLength(2);
    expect(out.every(i => i.kind === 'message')).toBe(true);
  });

  it('folds consecutive same-step messages into one group', () => {
    const out = groupMessagesBySteps([
      assistant(step(0)),
      assistant(step(0)),
      assistant(step(0)),
    ]);
    expect(out).toHaveLength(1);
    expect(out[0].kind).toBe('stepGroup');
    if (out[0].kind === 'stepGroup') {
      expect(out[0].messages).toHaveLength(3);
      expect(out[0].key).toBe('p1:0');
    }
  });

  it('splits into separate groups when the subtask changes', () => {
    const out = groupMessagesBySteps([
      assistant(step(0)),
      assistant(step(1)),
    ]);
    expect(out).toHaveLength(2);
    expect(out[0].kind).toBe('stepGroup');
    expect(out[1].kind).toBe('stepGroup');
  });

  it('breaks a group when a standalone message interrupts the same step', () => {
    // A synthesis message (no planStep) between two step-0 messages must not
    // merge them back into one group.
    const out = groupMessagesBySteps([
      assistant(step(0)),
      assistant(),
      assistant(step(0)),
    ]);
    expect(out.map(i => i.kind)).toEqual(['stepGroup', 'message', 'stepGroup']);
  });

  it('keys groups by planId + subtaskId across different plans', () => {
    const out = groupMessagesBySteps([
      assistant(step(0, 'p1')),
      assistant(step(0, 'p2')),
    ]);
    expect(out).toHaveLength(2);
    if (out[0].kind === 'stepGroup' && out[1].kind === 'stepGroup') {
      expect(out[0].key).toBe('p1:0');
      expect(out[1].key).toBe('p2:0');
    }
  });
});
