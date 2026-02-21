# Phase 13.1 Implementation Plan: Chat Tab Enhancement

**Date**: 2026-02-21
**Scope**: Enhance existing ChatTab with missing features from spec
**Estimated Complexity**: Medium (enhancement, not greenfield)

---

## Current State Assessment

The ChatTab (`client/src/components/tabs/ChatTab.tsx`, 584 lines) is **already substantially implemented**:

### Already Functional
- [x] Auto-growing textarea with Enter to submit (Shift+Enter for newline)
- [x] Streaming SSE responses with typing indicator
- [x] Markdown rendering with syntax highlighting (via MessageContent)
- [x] Model selector dropdown with provider grouping
- [x] Prompt profiles with descriptions
- [x] Token tracking per message
- [x] Reasoning trace toggle (show/hide thinking steps)
- [x] ThinkingBubble component for extended thinking
- [x] Session management (sessionId tracking)
- [x] Abort/cancel streaming capability

### Not Yet Implemented (from Phase 13.1 spec)
- [ ] Character/token counter in input area
- [ ] Paste image support (vision models)
- [ ] Voice input button (placeholder)
- [ ] Copy button per message
- [ ] Regenerate button on assistant messages
- [ ] Timestamp on hover
- [ ] Temperature slider (collapsed by default)
- [ ] "Use memory" toggle
- [ ] New chat button
- [ ] Recent chats list (last 10)
- [ ] "Save to Agent" button
- [ ] Compact, centered layout (max-width ~800px)

---

## Preliminary Work Required

### 1. Stack Assessment: No Major Changes Needed

The current stack fully supports Phase 13.1:
- **React 19 + TypeScript + Vite**: All in place
- **Lucide Icons**: Already used throughout
- **CSS Variables**: Design system established
- **API Client**: Streaming and chat endpoints ready
- **ServerContext**: State management ready

### 2. Minor Additions Needed

#### 2.1 localStorage Schema for Recent Chats
```typescript
// New storage keys (in lib/storage.ts)
agentx:server:{serverId}:recentChats  // Array<RecentChat>
agentx:server:{serverId}:savedChats   // Array<SavedChat> (for "Save to Agent")
```

#### 2.2 Types Addition (in lib/api.ts or new types.ts)
```typescript
interface RecentChat {
  id: string;
  sessionId: string;
  title: string;           // Auto-generated from first message
  preview: string;         // First ~50 chars of first message
  messageCount: number;
  createdAt: string;
  lastMessageAt: string;
}
```

---

## Implementation Plan

### Step 1: Input Area Enhancements

**File**: `ChatTab.tsx` (modify existing)

#### 1.1 Character/Token Counter
- Add counter below textarea: `{chars} chars · ~{tokens} tokens`
- Token estimation: `Math.ceil(chars / 4)` (rough heuristic)
- Subtle styling: `opacity: 0.5`, small font
- Show in amber when approaching model limit

#### 1.2 Temperature Slider (Collapsed)
- Add collapsible "Advanced" section below model selector
- Temperature slider: 0.0 - 2.0, default 0.7
- Store in component state, pass to API

#### 1.3 "Use Memory" Toggle
- Simple toggle switch next to model selector
- Default: ON
- Pass `use_memory: boolean` to chat request
- Requires backend support (already exists via AgentConfig.enable_memory)

**Estimated effort**: 2-3 hours

---

### Step 2: Message Enhancements

**Files**: `ChatTab.tsx`, `MessageContent.tsx`, new `MessageActions.tsx`

#### 2.1 Copy Button per Message
- Add copy icon button to message hover actions
- Copy raw content (not rendered markdown)
- Toast notification: "Copied to clipboard"

#### 2.2 Regenerate Button
- Add regenerate icon on assistant messages
- Regenerate: remove last assistant message, re-send last user message
- Preserve conversation branch (don't modify history)

#### 2.3 Timestamp on Hover
- Show relative timestamp on message hover
- Format: "2 min ago", "1 hour ago", "Yesterday"
- Use `date-fns` or simple relative time function

#### 2.4 Message Actions Component
```tsx
// New: components/chat/MessageActions.tsx
interface MessageActionsProps {
  message: Message;
  onCopy: () => void;
  onRegenerate?: () => void;  // Only for assistant
}
```

**Estimated effort**: 2-3 hours

---

### Step 3: Session Management UI

**Files**: `ChatTab.tsx`, new `ChatSidebar.tsx` or inline

#### 3.1 New Chat Button
- Add "New Chat" button in header area
- Clears messages, generates new sessionId
- Optional: prompt to save current chat

#### 3.2 Recent Chats List
- Collapsible sidebar or dropdown
- Shows last 10 chats (localStorage)
- Display: title, preview, timestamp
- Click to restore session
- Delete button per item

#### 3.3 Chat Persistence (localStorage)
- Save on: new message, tab close
- Structure:
  ```typescript
  {
    sessionId: string,
    messages: Message[],
    model: string,
    createdAt: string,
    title: string  // Auto from first user message
  }
  ```

#### 3.4 "Save to Agent" Button
- Appears when chat has messages
- Promotes chat to Agent tab conversation
- Opens Agent tab with conversation loaded
- Requires Agent tab conversation support (Phase 13.2 dependency)

**Estimated effort**: 4-5 hours

---

### Step 4: Layout Refinement

**Files**: `ChatTab.css`, `ChatTab.tsx`

#### 4.1 Compact Centered Layout
- Max-width: 800px for message area
- Center horizontally with `margin: 0 auto`
- Keep sidebar/header full width
- Responsive: full width on mobile

#### 4.2 Styling Polish
- Ensure consistent spacing
- Subtle dividers between features
- Hover states for all interactive elements

**Estimated effort**: 1-2 hours

---

### Step 5: Future Placeholders

#### 5.1 Voice Input Button (Placeholder)
- Add microphone icon button (disabled)
- Tooltip: "Voice input coming soon"
- No functionality in 13.1

#### 5.2 Image Paste Support (Placeholder)
- Detect paste event with image
- Show toast: "Image support coming soon"
- No functionality in 13.1

**Estimated effort**: 30 minutes

---

## API Considerations

### Existing Endpoints (No Changes Needed for 13.1)
- `POST /api/agent/chat` - Already supports `session_id`, `model`, `show_reasoning`
- `POST /api/agent/chat/stream` - SSE streaming ready

### Backend Changes Required

**Temperature parameter** is currently hardcoded at `0.7` in `views.py:503`:
```python
async for chunk in provider.stream(messages, model_id, temperature=0.7, max_tokens=2000)
```

Changes needed in `views.py`:
1. Accept `temperature` from request body (default: 0.7)
2. Accept `use_memory` from request body (default: true)
3. Pass `temperature` to `provider.stream()` and `agent.chat()`

```python
# views.agent_chat and views.agent_chat_stream updates:
temperature = data.get("temperature", 0.7)
use_memory = data.get("use_memory", True)

# Pass to provider.stream():
async for chunk in provider.stream(messages, model_id, temperature=temperature, max_tokens=2000)

# For agent.chat(), pass via AgentConfig:
agent = Agent(AgentConfig(default_model=model, enable_memory=use_memory))
```

This is a minor backend change (~10 lines) that should be done as part of Step 1.

---

## File Changes Summary

| File | Action | Description |
|------|--------|-------------|
| `ChatTab.tsx` | Modify | Add new features, state management |
| `ChatTab.css` | Modify | Layout changes, new component styles |
| `MessageContent.tsx` | Modify | Add copy/action support |
| `MessageActions.tsx` | Create | New component for message actions |
| `ChatSidebar.tsx` | Create | Recent chats list (optional, could be inline) |
| `lib/storage.ts` | Modify | Add recent chats storage functions |
| `lib/api.ts` | Modify | Add temperature/memory params |
| `lib/hooks.ts` | Modify | Add useRecentChats hook |

---

## Dependencies

### External Packages (Already Installed)
- `lucide-react` - Icons (Copy, RefreshCw, Mic, etc.)
- `react-markdown` - Already used
- `date-fns` - May need to add for relative time

### Internal Dependencies
- ServerContext - For per-server chat storage
- API client - For chat requests
- MessageContent - For rendering

---

## Testing Checklist

- [ ] Character counter updates on input
- [ ] Token estimate is reasonable
- [ ] Temperature slider affects response style
- [ ] Memory toggle works (when backend supports it)
- [ ] Copy button copies correct content
- [ ] Regenerate produces new response
- [ ] Timestamps show correctly on hover
- [ ] New chat clears conversation
- [ ] Recent chats persist across refresh
- [ ] Recent chats restore correctly
- [ ] Layout is centered and responsive
- [ ] All hover states work
- [ ] Keyboard navigation works

---

## Out of Scope (Deferred to 13.2+)

- Conversation branching (13.2)
- Full conversation persistence to database (13.5)
- Agent profiles (13.3)
- "Save to Agent" full implementation (requires 13.2)
- Image upload and vision model support
- Voice input implementation

---

## Recommended Implementation Order

1. **Input enhancements** (counter, temperature) - Low risk, immediate value
2. **Message actions** (copy, timestamp) - High value, moderate complexity
3. **Layout refinement** - Quick win, visual polish
4. **Session management** (new chat, recent chats) - Higher complexity
5. **Regenerate button** - Requires careful state management
6. **Placeholders** - Trivial, do last

---

## Success Criteria

Phase 13.1 is complete when:
1. User can see character/token count while typing
2. User can adjust temperature via slider
3. User can toggle memory on/off
4. User can copy any message content
5. User can regenerate assistant responses
6. User sees timestamps on message hover
7. User can start a new chat
8. User can see and restore recent chats (last 10)
9. Layout is centered with 800px max-width
10. All interactions feel smooth and responsive
