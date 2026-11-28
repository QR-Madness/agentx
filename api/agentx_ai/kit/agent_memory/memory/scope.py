"""Row-level-security scopes for the PostgreSQL memory tables (Decisions.md ADR-21).

Every ``memory_*`` table forces row-level security (Alembic ``0010``): a
transaction sees one user's rows after it has named that user, every user's rows
as the system role, and nothing otherwise. This module is the only place either
is entered, which is what lets a test hold the line
(``tests_memory.MemoryScopeBoundaryTest``):

- ``user_scope(user_id)``: the scope every memory class works in.
- ``system_scope()``: every user's rows. For ``CrossUserStore`` and nothing else.

Both drop from the login role to a role that cannot log in, with ``SET LOCAL
ROLE``. That step is not optional: the bundled stack logs in as a superuser, and
superusers bypass row-level security even when it is forced. ``SET LOCAL`` ends
with the transaction, so a pooled connection goes back clean; a scope therefore
re-enters at the start of every transaction its session begins, and a commit or
rollback inside the ``with`` block does not step outside it.

The PostgreSQL twins of the memory classes (Slice 3) are built on these; until
then nothing reads the tables.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

from sqlalchemy import event, text
from sqlalchemy.engine import Connection
from sqlalchemy.orm import Session

from ..connections import get_postgres_session

# The roles and the setting, as Alembic 0010 creates and reads them.
MEMORY_ROLE = "agentx_memory"
SYSTEM_ROLE = "agentx_memory_system"
USER_SETTING = "agentx.user_id"

# Every table that forces row-level security, in dependency order. A live test
# compares this with the database, so a memory table added without its policies,
# or without a line here, fails there.
MEMORY_TABLES: tuple[str, ...] = (
    "memory_conversations",
    "memory_turns",
    "memory_entities",
    "memory_facts",
    "memory_goals",
    "memory_strategies",
    "memory_procedures",
    "memory_tools",
    "memory_tool_invocations",
    "memory_edges",
)

# Static statements: a role name is an identifier and cannot be a bound parameter.
_SET_MEMORY_ROLE = text("SET LOCAL ROLE agentx_memory")
_SET_SYSTEM_ROLE = text("SET LOCAL ROLE agentx_memory_system")
_NAME_USER = text("SELECT set_config('agentx.user_id', :user_id, true)")


def _require_user(user_id: str) -> None:
    """Refuse an empty id: it would name nobody, and every query after it would
    quietly return nothing."""
    if not isinstance(user_id, str) or not user_id.strip():
        raise ValueError("a memory scope needs a user id")


def enter_user_scope(executor: Session | Connection, user_id: str) -> None:
    """Confine the current transaction of `executor` to `user_id`'s memory rows.

    It lasts for that one transaction. Prefer ``user_scope``, which re-enters for
    every transaction of its session.
    """
    _require_user(user_id)
    executor.execute(_SET_MEMORY_ROLE)
    executor.execute(_NAME_USER, {"user_id": user_id})


def enter_system_scope(executor: Session | Connection) -> None:
    """Open every user's memory rows to the current transaction of `executor`.

    For work that legitimately crosses users. Call it from ``CrossUserStore`` only.
    """
    executor.execute(_SET_SYSTEM_ROLE)


def _enter_every_transaction(session: Session, enter: Callable[[Connection], None]) -> None:
    """Run `enter` at the start of each transaction `session` begins, so that a
    commit or rollback part-way through leaves the next statement in scope too."""

    def on_begin(_session: Session, _transaction: Any, connection: Connection) -> None:
        enter(connection)

    event.listen(session, "after_begin", on_begin)


@contextmanager
def user_scope(user_id: str) -> Iterator[Session]:
    """A session confined to `user_id`'s memory rows, in every transaction it
    begins. Commits on a clean exit and rolls back on an exception."""
    _require_user(user_id)
    with get_postgres_session() as session:
        _enter_every_transaction(session, lambda connection: enter_user_scope(connection, user_id))
        yield session


@contextmanager
def system_scope() -> Iterator[Session]:
    """A session that sees every user's memory rows, in every transaction it
    begins. For ``CrossUserStore`` only. Commits on a clean exit, rolls back on an
    exception."""
    with get_postgres_session() as session:
        _enter_every_transaction(session, enter_system_scope)
        yield session
