"""Durable connection incarnations and native backend session ids.

Each connected-agent name has a persisted incarnation. A consult captures
that incarnation before calling the backend and may publish the returned
session id only while the same connection identity remains active. Deleting
and recreating a connection therefore cannot let a late result resurrect the
old backend session, even when the name is reused.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
import sys
import threading
from typing import Any, BinaryIO, Iterator
import uuid

from deeptutor.services.path_service import get_path_service

logger = logging.getLogger(__name__)

_FILE = "subagent_sessions.json"
_SEP = "::"
_VERSION = 2
_thread_guard = threading.RLock()


class ConnectionIncarnationChanged(RuntimeError):
    """A connected-agent pointer was deleted or replaced during a consult."""


@dataclass(frozen=True, slots=True)
class ConnectionIncarnation:
    """Persisted identity captured by one connected-agent consult."""

    connection: str
    kind: str
    cwd: str
    value: str


def session_key(chat_session_id: str, connection: str) -> str:
    """Return the registry key for one chat and connection pair."""

    return f"{chat_session_id}{_SEP}{connection}"


def _path() -> Path:
    return get_path_service().get_settings_file(_FILE)


def _lock_path() -> Path:
    path = _path()
    return path.with_name(f".{path.name}.lock")


def _lock(handle: BinaryIO) -> None:
    if sys.platform == "win32":
        import msvcrt

        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        msvcrt.locking(  # type: ignore[attr-defined]
            handle.fileno(),
            msvcrt.LK_LOCK,  # type: ignore[attr-defined]
            1,
        )
        return

    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)


def _unlock(handle: BinaryIO) -> None:
    if sys.platform == "win32":
        import msvcrt

        handle.seek(0)
        msvcrt.locking(  # type: ignore[attr-defined]
            handle.fileno(),
            msvcrt.LK_UNLCK,  # type: ignore[attr-defined]
            1,
        )
        return

    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def _registry_lock() -> Iterator[None]:
    """Serialize registry mutations across threads and local processes."""

    with _thread_guard:
        lock_path = _lock_path()
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+b") as handle:
            _lock(handle)
            try:
                yield
            finally:
                _unlock(handle)


def _empty_registry() -> dict[str, Any]:
    return {"version": _VERSION, "connections": {}, "sessions": {}}


def _load_unlocked() -> dict[str, Any]:
    path = _path()
    if not path.exists():
        return _empty_registry()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.warning("failed to read %s; starting an empty session registry", path, exc_info=True)
        return _empty_registry()
    if not isinstance(raw, dict):
        return _empty_registry()
    if raw.get("version") == _VERSION:
        connections = raw.get("connections")
        sessions = raw.get("sessions")
        if isinstance(connections, dict) and isinstance(sessions, dict):
            return {
                "version": _VERSION,
                "connections": dict(connections),
                "sessions": dict(sessions),
            }
        return _empty_registry()

    # Version 1 stored session entries at the root. They stay unavailable
    # until a live connection claims matching kind/cwd values below.
    sessions = {key: value for key, value in raw.items() if isinstance(value, dict)}
    return {"version": _VERSION, "connections": {}, "sessions": sessions}


def _save_unlocked(data: dict[str, Any]) -> None:
    path = _path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _connection_entry(data: dict[str, Any], connection: str) -> dict[str, Any] | None:
    entry = data["connections"].get(connection)
    return entry if isinstance(entry, dict) else None


def _entry_matches(entry: dict[str, Any], snapshot: ConnectionIncarnation) -> bool:
    return (
        bool(entry.get("active"))
        and str(entry.get("kind") or "") == snapshot.kind
        and str(entry.get("cwd") or "") == snapshot.cwd
        and str(entry.get("incarnation") or "") == snapshot.value
    )


def _purge_connection_sessions(data: dict[str, Any], connection: str) -> None:
    suffix = f"{_SEP}{connection}"
    for key in [key for key in data["sessions"] if key.endswith(suffix)]:
        data["sessions"].pop(key, None)


def activate_connection(connection: str, *, kind: str, cwd: str) -> ConnectionIncarnation:
    """Create a fresh persisted identity for a newly connected pointer."""

    snapshot = ConnectionIncarnation(
        connection=str(connection or "").strip(),
        kind=str(kind or "").strip(),
        cwd=str(cwd or "").strip(),
        value=uuid.uuid4().hex,
    )
    if not snapshot.connection or not snapshot.kind:
        raise ValueError("connection and kind are required")
    with _registry_lock():
        data = _load_unlocked()
        _purge_connection_sessions(data, snapshot.connection)
        data["connections"][snapshot.connection] = {
            "active": True,
            "kind": snapshot.kind,
            "cwd": snapshot.cwd,
            "incarnation": snapshot.value,
        }
        _save_unlocked(data)
    return snapshot


def connection_incarnation(connection: str, *, kind: str, cwd: str) -> ConnectionIncarnation:
    """Return the active identity, adopting a matching legacy pointer once."""

    normalized_connection = str(connection or "").strip()
    normalized_kind = str(kind or "").strip()
    normalized_cwd = str(cwd or "").strip()
    if not normalized_connection or not normalized_kind:
        raise ConnectionIncarnationChanged("Connected Agent identity is incomplete.")

    with _registry_lock():
        data = _load_unlocked()
        entry = _connection_entry(data, normalized_connection)
        if entry is None:
            snapshot = ConnectionIncarnation(
                normalized_connection,
                normalized_kind,
                normalized_cwd,
                uuid.uuid4().hex,
            )
            data["connections"][normalized_connection] = {
                "active": True,
                "kind": snapshot.kind,
                "cwd": snapshot.cwd,
                "incarnation": snapshot.value,
            }
            suffix = f"{_SEP}{normalized_connection}"
            for key, session in list(data["sessions"].items()):
                if not key.endswith(suffix) or not isinstance(session, dict):
                    continue
                if (
                    str(session.get("kind") or "") == snapshot.kind
                    and str(session.get("cwd") or "") == snapshot.cwd
                ):
                    session["incarnation"] = snapshot.value
                else:
                    data["sessions"].pop(key, None)
            _save_unlocked(data)
            return snapshot

        snapshot = ConnectionIncarnation(
            normalized_connection,
            normalized_kind,
            normalized_cwd,
            str(entry.get("incarnation") or ""),
        )
        if not snapshot.value or not _entry_matches(entry, snapshot):
            raise ConnectionIncarnationChanged(
                f"Connected Agent {normalized_connection!r} was deleted or replaced."
            )
        return snapshot


def revoke_connection(connection: str) -> None:
    """Invalidate one connection and every session pointer owned by it."""

    normalized = str(connection or "").strip()
    if not normalized:
        return
    with _registry_lock():
        data = _load_unlocked()
        previous = _connection_entry(data, normalized) or {}
        data["connections"][normalized] = {
            "active": False,
            "kind": str(previous.get("kind") or ""),
            "cwd": str(previous.get("cwd") or ""),
            "incarnation": uuid.uuid4().hex,
        }
        _purge_connection_sessions(data, normalized)
        _save_unlocked(data)


def get_session(key: str, *, incarnation: ConnectionIncarnation) -> str | None:
    """Return a session only when connection and stored identities still match."""

    with _registry_lock():
        data = _load_unlocked()
        connection = _connection_entry(data, incarnation.connection)
        if connection is None or not _entry_matches(connection, incarnation):
            return None
        entry = data["sessions"].get(key)
        if not isinstance(entry, dict):
            return None
        if (
            str(entry.get("kind") or "") != incarnation.kind
            or str(entry.get("cwd") or "") != incarnation.cwd
            or str(entry.get("incarnation") or "") != incarnation.value
        ):
            return None
        session_id = str(entry.get("session_id") or "")
        return session_id or None


def remember_session(
    key: str,
    session_id: str,
    *,
    incarnation: ConnectionIncarnation,
) -> bool:
    """CAS-publish a native session id for the captured connection identity."""

    if not session_id:
        return False
    with _registry_lock():
        data = _load_unlocked()
        connection = _connection_entry(data, incarnation.connection)
        if connection is None or not _entry_matches(connection, incarnation):
            return False
        data["sessions"][key] = {
            "session_id": session_id,
            "kind": incarnation.kind,
            "cwd": incarnation.cwd,
            "incarnation": incarnation.value,
        }
        _save_unlocked(data)
        return True


def assert_connection_current(incarnation: ConnectionIncarnation) -> None:
    """Raise when a consult's captured identity is no longer active."""

    with _registry_lock():
        data = _load_unlocked()
        connection = _connection_entry(data, incarnation.connection)
        if connection is None or not _entry_matches(connection, incarnation):
            raise ConnectionIncarnationChanged(
                f"Connected Agent {incarnation.connection!r} was deleted or replaced."
            )


__all__ = [
    "ConnectionIncarnation",
    "ConnectionIncarnationChanged",
    "activate_connection",
    "assert_connection_current",
    "connection_incarnation",
    "get_session",
    "remember_session",
    "revoke_connection",
    "session_key",
]
