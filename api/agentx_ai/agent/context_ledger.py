"""Context Ledger — priority-based budget allocator for the turn preamble.

The shipped ``assemble_turn_context`` fits only the verbatim transcript and keeps
*every* system block by fiat. As the preamble grows (checkpoints, scratchpad,
summary, memory, participants, reflex core, the stable salient core, soon the
workspace manifest) those blocks silently crowd out the transcript.

The ledger fixes that: each contributor registers a :class:`LedgerBlock` with a
``priority`` and an optional ``shrink_fn``; one :func:`assemble_ledger` allocator
decides what fits the token budget — keeping high-priority blocks at full size,
gracefully shrinking mid-priority ones, and dropping the lowest-priority blocks
that can't meet their ``min_tokens``. Per-block accounting is returned in the
:class:`LedgerResult` (the data seam a future Context Inspector reads).

Back-compat: ``mandatory`` blocks are always emitted full (the old "keep every
system block by fiat" behaviour). ``assemble_turn_context`` becomes a thin wrapper
that registers every system block as mandatory, so existing callers are
byte-identical. Allocation order is priority-driven; **emission** order preserves
the caller's registration order so the rendered prompt looks the same.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

from ..providers.base import Message, MessageRole

logger = logging.getLogger(__name__)


# Rough char→token estimate, kept identical to ContextManager.estimate_tokens so
# the ledger and the legacy assembler agree. (tiktoken consolidation is Foundation #6.)
def estimate_text_tokens(text: str) -> int:
    """Estimate tokens for a raw string (pre-``Message``). ~4 chars/token + overhead."""
    return (len(text) // 4) + 10


@dataclass
class LedgerBlock:
    """A preamble contributor competing for the token budget.

    ``content`` is rendered eagerly by the caller (``None``/empty → skipped, status
    ``empty``). ``priority`` drives *allocation* order (higher kept first); emission
    order is the caller's registration order. ``shrink_fn(content, target_tokens)``
    returns a smaller rendering for graceful degradation. ``mandatory`` blocks are
    always emitted full and never dropped/shrunk (back-compat "by fiat").
    """

    key: str
    priority: int
    content: str | None = None
    min_tokens: int = 0
    max_tokens: int | None = None
    shrink_fn: Callable[[str, int], str] | None = None
    mandatory: bool = False
    role: MessageRole = MessageRole.SYSTEM


@dataclass
class BlockAllocation:
    """Per-block accounting — the Context Inspector's data seam."""

    key: str
    priority: int
    requested_tokens: int
    granted_tokens: int
    status: str  # full | shrunk | dropped | empty


@dataclass
class LedgerResult:
    """Outcome of one allocation pass."""

    messages: list[Message]
    allocations: list[BlockAllocation] = field(default_factory=list)
    history_kept: int = 0
    history_dropped: int = 0
    input_budget: int = 0
    used_tokens: int = 0


_HISTORY_KEY = "__history__"


def _estimate_messages(messages: list[Message]) -> int:
    """Token estimate for a list of messages (mirrors ContextManager.estimate_tokens)."""
    total_chars = sum(len(m.content) for m in messages)
    return (total_chars // 4) + len(messages) * 10


def fit_history(
    history: list[Message],
    budget: int,
    recent_floor: int,
) -> tuple[list[Message], int]:
    """Fit recent verbatim history newest→oldest into ``budget`` tokens.

    Always keeps at least ``recent_floor`` of the most recent turns even under
    pressure. Returns ``(fitted, used_tokens)``. Factored out of the legacy
    ``assemble_turn_context`` loop so both paths share one implementation.
    """
    if budget <= 0:
        floor = history[-recent_floor:] if recent_floor > 0 else []
        return floor, _estimate_messages(floor)

    fitted: list[Message] = []
    used = 0
    for msg in reversed(history):  # newest → oldest
        tokens = _estimate_messages([msg])
        if fitted and used + tokens > budget and len(fitted) >= recent_floor:
            break
        fitted.insert(0, msg)
        used += tokens
    return fitted, used


def assemble_ledger(
    *,
    blocks: list[LedgerBlock],
    history: list[Message],
    new_message: Message,
    context_window: int,
    reserved_tokens: int,
    verbatim_ratio: float = 0.7,
    recent_floor: int = 4,
    history_priority: int = 50,
) -> LedgerResult:
    """Allocate the turn preamble + transcript within the token budget by priority.

    ``input_budget = min(verbatim_ratio · window, window − reserved_tokens)``. The
    ``new_message`` is mandatory and never dropped. Mandatory blocks are emitted full
    first; the remaining blocks (plus a synthetic history entry at ``history_priority``)
    compete by priority — fit full, else ``shrink_fn`` if ``remaining ≥ min_tokens``,
    else drop. Output ordering: ``kept blocks (registration order) + fitted_history +
    [new_message]``.
    """
    input_budget = min(
        int(context_window * verbatim_ratio),
        context_window - reserved_tokens,
    )
    new_tokens = _estimate_messages([new_message])
    remaining = input_budget - new_tokens

    allocations: dict[str, BlockAllocation] = {}
    granted_content: dict[str, str] = {}

    # 1. Mandatory blocks first — always emitted full, never dropped (back-compat
    #    "by fiat"). Emitted even when content is empty so the legacy assembler's
    #    "system_blocks + …" output is reproduced exactly.
    for b in blocks:
        if not b.mandatory:
            continue
        content = b.content or ""
        want = estimate_text_tokens(content)
        granted_content[b.key] = content
        allocations[b.key] = BlockAllocation(b.key, b.priority, want, want, "full")
        remaining -= want

    # 2. The rest compete by priority (a synthetic history entry rides along so the
    #    transcript yields to higher-priority blocks and outranks lower ones).
    optional = [b for b in blocks if not b.mandatory]
    # Stable sort: priority DESC, registration order breaks ties.
    indexed = sorted(
        enumerate(optional), key=lambda it: (-it[1].priority, it[0])
    )

    fitted_history: list[Message] = []
    history_used = 0
    history_done = False

    def _allocate_history() -> None:
        nonlocal fitted_history, history_used, history_done, remaining
        if history_done:
            return
        fitted_history, history_used = fit_history(
            history, max(0, remaining), recent_floor
        )
        remaining -= history_used
        history_done = True

    for _, b in indexed:
        # History slots in at its priority position relative to the optional blocks.
        if not history_done and b.priority < history_priority:
            _allocate_history()

        if not b.content:
            allocations[b.key] = BlockAllocation(b.key, b.priority, 0, 0, "empty")
            continue

        want = estimate_text_tokens(b.content)
        if b.max_tokens is not None:
            want = min(want, b.max_tokens)

        if want <= remaining:
            granted_content[b.key] = b.content
            allocations[b.key] = BlockAllocation(b.key, b.priority, want, want, "full")
            remaining -= want
        elif b.shrink_fn is not None and remaining >= b.min_tokens:
            shrunk = b.shrink_fn(b.content, remaining)
            stoks = estimate_text_tokens(shrunk)
            if shrunk and stoks <= remaining and stoks >= b.min_tokens:
                granted_content[b.key] = shrunk
                allocations[b.key] = BlockAllocation(
                    b.key, b.priority, want, stoks, "shrunk"
                )
                remaining -= stoks
            else:
                allocations[b.key] = BlockAllocation(b.key, b.priority, want, 0, "dropped")
        else:
            allocations[b.key] = BlockAllocation(b.key, b.priority, want, 0, "dropped")

    # History may still be pending if every optional block outranked it.
    _allocate_history()

    # 3. Emit in canonical (registration) order, then history, then the new message.
    messages: list[Message] = []
    for b in blocks:
        if b.key in granted_content:
            messages.append(Message(role=b.role, content=granted_content[b.key]))
    messages += fitted_history
    messages.append(new_message)

    used_tokens = sum(a.granted_tokens for a in allocations.values()) + history_used + new_tokens
    return LedgerResult(
        messages=messages,
        allocations=[allocations[b.key] for b in blocks if b.key in allocations],
        history_kept=len(fitted_history),
        history_dropped=max(0, len(history) - len(fitted_history)),
        input_budget=input_budget,
        used_tokens=used_tokens,
    )


# --- Shrink helpers ---------------------------------------------------------

_MARKER = "\n…[truncated]"


def shrink_tail(content: str, target_tokens: int) -> str:
    """Keep the head, drop the tail, so the result estimates to ≤ ``target_tokens``.

    ``estimate_text_tokens`` adds a fixed +10 overhead and the truncation marker
    costs a few chars, so the char budget is ``(target − overhead) · 4`` — without
    this margin the shrunk text re-estimates just over budget and the allocator
    would reject every shrink.
    """
    budget_chars = max(0, (target_tokens - 10) * 4 - len(_MARKER))
    if len(content) <= budget_chars or len(content) // 4 + 10 <= target_tokens:
        return content
    return content[:budget_chars].rstrip() + _MARKER


def shrink_lines_newest_n(content: str, target_tokens: int) -> str:
    """Keep the most recent lines (tail) that fit ``target_tokens``.

    For append-style blocks (checkpoints) where the newest entries matter most.
    The first line is treated as a header and always kept.
    """
    lines = content.split("\n")
    if not lines:
        return content
    header, body = lines[0], lines[1:]
    kept: list[str] = []
    used = estimate_text_tokens(header)
    for line in reversed(body):  # newest → oldest
        t = estimate_text_tokens(line)
        if used + t > target_tokens and kept:
            break
        kept.insert(0, line)
        used += t
    return "\n".join([header, *kept])


def shrink_memory_to_facts(content: str, target_tokens: int) -> str:
    """Degrade a rendered memory bundle to its ``## Known Facts`` section only.

    Drops ``## Relevant Entities`` / ``## Active Goals`` / ``## Relevant Past
    Conversations`` so the highest-signal facts survive budget pressure. If a
    header line precedes the sections (e.g. "Stable context …:") it is preserved.
    """
    if "## Known Facts" not in content:
        return shrink_tail(content, target_tokens)
    head, _, rest = content.partition("## Known Facts")
    # Keep any leading wrapper line (before the first section) + the facts section
    # up to the next "## " header.
    facts_body = rest.split("\n## ", 1)[0]
    prefix = head.split("##", 1)[0].rstrip()
    facts = f"## Known Facts{facts_body}".rstrip()
    out = f"{prefix}\n{facts}" if prefix else facts
    return shrink_tail(out, target_tokens) if target_tokens > 0 else out


# --- Dedup ------------------------------------------------------------------

def dedup_recall_against_core(recall_bundle, core_bundle) -> None:
    """Drop facts/entities from ``recall_bundle`` already present in ``core_bundle``.

    Mutates ``recall_bundle`` in place. Both bundles project an ``id`` on each fact
    and entity (the Cypher in semantic.py returns ``f.id``/``e.id``), so dedup is by
    id. Safe no-op when either bundle is falsy or lacks the lists.
    """
    if not recall_bundle or not core_bundle:
        return
    core_fact_ids = {f.get("id") for f in (core_bundle.facts or []) if f.get("id")}
    core_entity_ids = {e.get("id") for e in (core_bundle.entities or []) if e.get("id")}
    if core_fact_ids and recall_bundle.facts:
        recall_bundle.facts = [
            f for f in recall_bundle.facts if f.get("id") not in core_fact_ids
        ]
    if core_entity_ids and recall_bundle.entities:
        recall_bundle.entities = [
            e for e in recall_bundle.entities if e.get("id") not in core_entity_ids
        ]
