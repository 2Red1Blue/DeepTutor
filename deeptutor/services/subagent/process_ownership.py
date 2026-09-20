"""Local process ownership for native Connected Agent execution."""

from __future__ import annotations

import atexit
import json
import os
from pathlib import Path
import socket
import threading
from typing import BinaryIO

from deeptutor.runtime.home import get_runtime_home
from deeptutor.utils.secret_files import ensure_private_directory, ensure_private_file


class ConnectedAgentProcessOwnershipError(RuntimeError):
    """Another local process owns native Connected Agent execution."""


class ConnectedAgentProcessOwnership:
    """Hold one non-blocking OS file lock until the owning process exits."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.pid = os.getpid()
        self._handle: BinaryIO | None = None

    def acquire(self) -> None:
        """Acquire the local execution lock or fail without waiting."""

        ensure_private_directory(self.path.parent)
        handle = self.path.open("a+b")
        ensure_private_file(self.path)
        try:
            self._lock(handle)
        except OSError as exc:
            owner = self._read_owner(handle)
            handle.close()
            suffix = f" Current owner: {owner}." if owner else ""
            raise ConnectedAgentProcessOwnershipError(
                "Connected Agent native execution is already owned by another "
                f"local DeepTutor process.{suffix}"
            ) from exc
        try:
            self._write_owner(handle)
        except OSError as exc:
            try:
                self._unlock(handle)
            finally:
                handle.close()
            raise ConnectedAgentProcessOwnershipError(
                "Connected Agent native execution ownership could not be recorded."
            ) from exc
        self._handle = handle

    def release(self) -> None:
        """Release a lock acquired by this process."""

        handle = self._handle
        self._handle = None
        if handle is None:
            return
        try:
            self._unlock(handle)
        finally:
            handle.close()

    def discard_inherited_handle(self) -> None:
        """Close a post-fork duplicate without unlocking the parent's lock."""

        handle = self._handle
        self._handle = None
        if handle is not None:
            handle.close()

    @staticmethod
    def _lock(handle: BinaryIO) -> None:
        if os.name == "nt":
            import msvcrt

            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            return

        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    @staticmethod
    def _unlock(handle: BinaryIO) -> None:
        if os.name == "nt":
            import msvcrt

            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            return

        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _write_owner(self, handle: BinaryIO) -> None:
        payload = json.dumps(
            {"hostname": socket.gethostname(), "pid": self.pid},
            sort_keys=True,
        ).encode("utf-8")
        handle.seek(0)
        handle.truncate()
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())

    @staticmethod
    def _read_owner(handle: BinaryIO) -> str:
        try:
            handle.seek(0)
            payload = json.loads(handle.read().decode("utf-8"))
        except (OSError, UnicodeDecodeError, ValueError, TypeError):
            return ""
        if not isinstance(payload, dict):
            return ""
        hostname = str(payload.get("hostname") or "").strip()
        pid = str(payload.get("pid") or "").strip()
        if hostname and pid:
            return f"{hostname}:{pid}"
        return pid or hostname


_ownership_guard = threading.Lock()
_process_ownership: ConnectedAgentProcessOwnership | None = None


def connected_agent_process_lock_path() -> Path:
    """Return the lock shared by API workers for one runtime home."""

    return get_runtime_home() / "data" / "user" / ".runtime" / "connected-agent-execution.lock"


def ensure_connected_agent_process_ownership() -> ConnectedAgentProcessOwnership:
    """Return this process's ownership, acquiring it on first native execution."""

    global _process_ownership
    pid = os.getpid()
    with _ownership_guard:
        current = _process_ownership
        if current is not None and current.pid == pid:
            return current
        if current is not None:
            current.discard_inherited_handle()
            _process_ownership = None
        ownership = ConnectedAgentProcessOwnership(connected_agent_process_lock_path())
        ownership.acquire()
        _process_ownership = ownership
        return ownership


def _release_process_ownership() -> None:
    global _process_ownership
    with _ownership_guard:
        current = _process_ownership
        _process_ownership = None
        if current is not None and current.pid == os.getpid():
            current.release()


atexit.register(_release_process_ownership)


__all__ = [
    "ConnectedAgentProcessOwnership",
    "ConnectedAgentProcessOwnershipError",
    "connected_agent_process_lock_path",
    "ensure_connected_agent_process_ownership",
]
