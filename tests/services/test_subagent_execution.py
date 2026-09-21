"""Concurrency contracts for shared Connected Agent execution."""

from __future__ import annotations

import asyncio

import pytest

from deeptutor.capabilities.subagent.tools import ConsultSubagentTool
from deeptutor.services.subagent.config import BackendConfig
from deeptutor.services.subagent.execution import consult_with_session_guard
from deeptutor.services.subagent.sessions import session_key
from deeptutor.services.subagent.types import ConsultResult


class _BlockingBackend:
    kind = "codex"
    local_cli = True

    def __init__(self) -> None:
        self.entered: asyncio.Queue[str | None] = asyncio.Queue()
        self.release = asyncio.Event()
        self.calls: list[str | None] = []

    async def consult(self, question, *, session_id, **_kwargs):
        self.calls.append(session_id)
        await self.entered.put(session_id)
        await self.release.wait()
        return ConsultResult(
            final_text=question,
            session_id=f"native-{len(self.calls)}",
            success=True,
        )


@pytest.fixture
def isolated_sessions(monkeypatch, tmp_path):
    from deeptutor.services.subagent import sessions

    monkeypatch.setattr(sessions, "_path", lambda: tmp_path / "sessions.json")


@pytest.mark.asyncio
async def test_sidebar_then_tool_resumes_session_returned_by_first_call(
    monkeypatch, isolated_sessions
) -> None:
    """The tool cannot read a stale session id while a sidebar run is active."""

    from deeptutor.services.subagent import access

    backend = _BlockingBackend()
    monkeypatch.setattr(access, "get_backend", lambda _kind: backend)
    monkeypatch.setattr(access, "_assert_single_worker_execution", lambda: None)
    monkeypatch.setattr(access, "subagent_backend_allowed", lambda _kind: True)
    monkeypatch.setattr(access, "_resolve_owner_workspace_cwd", lambda cwd: cwd)
    monkeypatch.setattr(
        access,
        "load_subagent_settings",
        lambda: type("Settings", (), {"backend": lambda self, _kind: BackendConfig()})(),
    )
    skey = session_key("chat-a", "Agent")

    async def noop_event(_event) -> None:
        return None

    sidebar = asyncio.create_task(
        consult_with_session_guard(
            backend,
            "sidebar",
            on_event=noop_event,
            cwd="/workspace",
            config=BackendConfig(),
            chat_session_id="chat-a",
            connection="Agent",
            session_key_value=skey,
        )
    )
    assert await asyncio.wait_for(backend.entered.get(), timeout=1) is None

    tool = asyncio.create_task(
        ConsultSubagentTool().execute(
            question="tool",
            _subagent={
                "kind": "codex",
                "cwd": "/workspace",
                "partner_id": "",
                "name": "Agent",
                "budget": 2,
                "config": BackendConfig(),
                "state": {"count": 0, "session_id": None, "name": "Agent"},
                "images": [],
                "session_key": skey,
                "chat_session_id": "chat-a",
                "connection": "Agent",
            },
        )
    )
    await asyncio.sleep(0)
    assert backend.calls == [None]
    backend.release.set()

    await sidebar
    result = await tool
    assert result.success is True
    assert backend.calls == [None, "native-1"]


@pytest.mark.asyncio
async def test_independent_chat_sessions_execute_in_parallel(isolated_sessions) -> None:
    backend = _BlockingBackend()

    async def noop_event(_event) -> None:
        return None

    tasks = [
        asyncio.create_task(
            consult_with_session_guard(
                backend,
                chat,
                on_event=noop_event,
                cwd="/workspace",
                config=BackendConfig(),
                chat_session_id=chat,
                connection="Agent",
                session_key_value=session_key(chat, "Agent"),
            )
        )
        for chat in ("chat-a", "chat-b")
    ]
    assert await asyncio.wait_for(backend.entered.get(), timeout=1) is None
    assert await asyncio.wait_for(backend.entered.get(), timeout=1) is None
    backend.release.set()
    await asyncio.gather(*tasks)


@pytest.mark.asyncio
async def test_cancelled_owner_releases_key_for_successor(isolated_sessions) -> None:
    backend = _BlockingBackend()

    async def noop_event(_event) -> None:
        return None

    kwargs = {
        "on_event": noop_event,
        "cwd": "/workspace",
        "config": BackendConfig(),
        "chat_session_id": "chat-a",
        "connection": "Agent",
        "session_key_value": session_key("chat-a", "Agent"),
    }
    owner = asyncio.create_task(consult_with_session_guard(backend, "owner", **kwargs))
    assert await asyncio.wait_for(backend.entered.get(), timeout=1) is None
    owner.cancel()
    with pytest.raises(asyncio.CancelledError):
        await owner

    successor = asyncio.create_task(consult_with_session_guard(backend, "successor", **kwargs))
    assert await asyncio.wait_for(backend.entered.get(), timeout=1) is None
    backend.release.set()
    result, key = await successor
    assert result.success is True
    assert key.owner_id
    assert key.backend_kind == "codex"
    assert key.chat_session_id == "chat-a"
    assert key.connection == "Agent"
    assert key.native_session_id == "native-2"
