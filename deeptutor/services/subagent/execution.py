"""Keyed execution for one native Connected Agent conversation."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
import threading
from typing import AsyncIterator

from deeptutor.multi_user.context import get_current_user
from deeptutor.services.subagent.base import OnEvent, SubagentBackend
from deeptutor.services.subagent.config import BackendConfig
from deeptutor.services.subagent.types import ConsultResult


@dataclass(frozen=True, slots=True)
class ExecutionKey:
    """Identity of one serialized native conversation."""

    owner_id: str
    backend_kind: str
    chat_session_id: str
    connection: str
    native_session_id: str
    connection_incarnation: str


@dataclass(slots=True)
class _LockEntry:
    lock: asyncio.Lock
    users: int = 0


_entries_guard = threading.Lock()
_entries: dict[tuple[str, str, str, str], _LockEntry] = {}


@asynccontextmanager
async def execution_guard(
    *, backend_kind: str, chat_session_id: str, connection: str
) -> AsyncIterator[tuple[str, str, str, str]]:
    """Serialize one account/backend/chat/connection execution sequence."""

    stable_key = (
        get_current_user().id,
        str(backend_kind or "").strip(),
        str(chat_session_id or "").strip(),
        str(connection or "").strip(),
    )
    with _entries_guard:
        entry = _entries.get(stable_key)
        if entry is None:
            entry = _LockEntry(asyncio.Lock())
            _entries[stable_key] = entry
        entry.users += 1
    try:
        async with entry.lock:
            yield stable_key
    finally:
        with _entries_guard:
            entry.users -= 1
            if entry.users == 0 and not entry.lock.locked():
                _entries.pop(stable_key, None)


async def consult_with_session_guard(
    backend: SubagentBackend,
    question: str,
    *,
    on_event: OnEvent,
    cwd: str,
    config: BackendConfig,
    chat_session_id: str,
    connection: str,
    session_key_value: str,
    state: dict | None = None,
    images: list[str] | None = None,
    partner_id: str | None = None,
) -> tuple[ConsultResult, ExecutionKey]:
    """Load, consult, and persist one backend session under the shared guard."""

    from deeptutor.services.subagent.sessions import (
        ConnectionIncarnationChanged,
        assert_connection_current,
        connection_incarnation,
        get_session,
        remember_session,
    )

    async with execution_guard(
        backend_kind=backend.kind,
        chat_session_id=chat_session_id,
        connection=connection,
    ) as stable_key:
        incarnation = connection_incarnation(connection, kind=backend.kind, cwd=cwd)
        persisted = (
            get_session(session_key_value, incarnation=incarnation) if session_key_value else None
        )
        state_incarnation = str((state or {}).get("connection_incarnation") or "")
        state_session_id = (
            str((state or {}).get("session_id") or "")
            if not session_key_value and state_incarnation == incarnation.value
            else ""
        )
        native_session_id = persisted or state_session_id or None
        result = await backend.consult(
            question,
            on_event=on_event,
            cwd=cwd or None,
            session_id=native_session_id,
            config=config,
            images=images,
            partner_id=partner_id,
        )
        if result.session_id:
            if session_key_value:
                committed = remember_session(
                    session_key_value,
                    result.session_id,
                    incarnation=incarnation,
                )
                if not committed:
                    raise ConnectionIncarnationChanged(
                        f"Connected Agent {connection!r} was deleted or replaced."
                    )
            else:
                assert_connection_current(incarnation)
            if state is not None:
                state["session_id"] = result.session_id
                state["connection_incarnation"] = incarnation.value
        else:
            assert_connection_current(incarnation)
        key = ExecutionKey(
            *stable_key,
            native_session_id or result.session_id or "",
            incarnation.value,
        )
        return result, key


__all__ = [
    "ExecutionKey",
    "consult_with_session_guard",
    "execution_guard",
]
