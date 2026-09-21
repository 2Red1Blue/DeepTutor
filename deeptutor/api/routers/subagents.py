"""Subagent connections API.

Backs the "My Agents → connected agents" feature: detect which local agent CLIs
or configured remote backends are usable, connect one as a pointer KB the chat
composer can select, and configure the consult budget. Connections are
stored as ``type: subagent`` knowledge bases (per-user, via the KB manager), so
they ride the same selection/persistence path as the other connected KB types —
the subagent capability drives them live, nothing is indexed.

Local CLIs run on the host with the host user's credentials; remote backends
use deployment-wide server configuration. Detection is therefore machine-
global. Unavailable runtimes are not offered by the connection UI.
"""

from __future__ import annotations

import asyncio
import json
import logging

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from deeptutor.api.routers.auth import require_admin
from deeptutor.knowledge.kb_types import SUBAGENT_KB_TYPE
from deeptutor.multi_user.context import get_current_user
from deeptutor.multi_user.knowledge_access import current_kb_manager
from deeptutor.services.subagent import (
    SubagentResolutionError,
    detect_all,
    executable_backend_kinds,
    list_backend_kinds,
    load_subagent_settings,
    native_execution_provenance,
    resolve_backend_execution,
    save_subagent_settings,
    settings_from_dict,
)

logger = logging.getLogger(__name__)
router = APIRouter()


class ConnectSubagentRequest(BaseModel):
    name: str
    agent_kind: str
    cwd: str = ""


class SubagentSettingsPayload(BaseModel):
    consult_budget: int | None = None
    backends: dict[str, dict] | None = Field(default=None)


class SubagentMessageRequest(BaseModel):
    chat_session_id: str = ""
    message: str


def _raise_resolution_error(exc: SubagentResolutionError) -> None:
    raise HTTPException(status_code=403 if exc.forbidden else 400, detail=exc.detail) from exc


@router.get("/detect")
async def detect_subagents():
    """Report which local and remote agent backends are usable."""
    detections = await detect_all(allowed_kinds=executable_backend_kinds())
    return {"backends": [d.to_dict() for d in detections]}


@router.get("/backends/options")
async def backend_options():
    """Synced model + reasoning-effort options per backend (settings page sync)."""
    from deeptutor.services.subagent.models import list_backend_options

    options = await list_backend_options(allowed_kinds=executable_backend_kinds())
    return {"backends": [o.to_dict() for o in options]}


@router.post("/backends/{kind}/sync")
async def sync_backend(kind: str):
    """Re-pull one backend's model catalog (the settings "sync" button).

    For Claude Code this scrapes its ``/model`` TUI live and caches the result;
    for Codex it re-reads the CLI-maintained cache.
    """
    from deeptutor.services.subagent.models import sync_backend_options

    try:
        resolved = resolve_backend_execution(kind, require_workspace=False)
    except SubagentResolutionError as exc:
        _raise_resolution_error(exc)
    if not resolved.backend.local_cli:
        # Only local CLIs have a model catalog to sync; partners run their own.
        raise HTTPException(status_code=400, detail=f"Unknown agent kind: {kind!r}")
    options = await sync_backend_options(kind)
    return options.to_dict()


@router.get("/connections")
async def list_connections():
    """List the current user's connected subagents."""
    manager = current_kb_manager()
    connections = []
    for name in manager.list_knowledge_bases():
        meta = manager.get_metadata(name)
        if not isinstance(meta, dict) or meta.get("type") != SUBAGENT_KB_TYPE:
            continue
        if meta.get("agent_kind") == "partner":
            continue
        connections.append(
            {
                "name": name,
                "agent_kind": meta.get("agent_kind", ""),
                "cwd": meta.get("cwd", ""),
                "partner_id": meta.get("partner_id", ""),
                "description": meta.get("description", ""),
                "created_at": meta.get("created_at"),
                "updated_at": meta.get("updated_at"),
                "execution_profile": "native",
                "provenance": native_execution_provenance(str(meta.get("agent_kind") or "")),
            }
        )
    return {"connections": connections}


@router.post("/connections")
async def create_connection(payload: ConnectSubagentRequest):
    """Connect a local or remote subagent as a selectable KB."""
    name = (payload.name or "").strip()
    agent_kind = (payload.agent_kind or "").strip()
    if not name or not agent_kind:
        raise HTTPException(status_code=400, detail="Both name and agent_kind are required.")
    if agent_kind not in list_backend_kinds():
        raise HTTPException(status_code=400, detail=f"Unknown agent kind: {agent_kind!r}")
    try:
        resolved = resolve_backend_execution(agent_kind, cwd=payload.cwd)
    except SubagentResolutionError as exc:
        _raise_resolution_error(exc)
    resolved_cwd = resolved.cwd

    try:
        manager = current_kb_manager()
        entry = manager.register_subagent_connection(name, agent_kind, cwd=resolved_cwd)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("Error connecting subagent: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "status": "connected",
        "name": name,
        "agent_kind": entry["agent_kind"],
        "cwd": entry["cwd"],
        "partner_id": entry.get("partner_id", ""),
        "execution_profile": "native",
        "provenance": native_execution_provenance(str(entry["agent_kind"])),
    }


@router.delete("/connections/{name}")
async def delete_connection(name: str):
    """Disconnect a subagent (removes the pointer KB; touches no files)."""
    manager = current_kb_manager()
    meta = manager.get_metadata(name)
    if not isinstance(meta, dict) or meta.get("type") != SUBAGENT_KB_TYPE:
        raise HTTPException(status_code=404, detail=f"No connected subagent named {name!r}.")
    try:
        manager.delete_knowledge_base(name, confirm=True)
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("Error disconnecting subagent: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    # Drop any remembered live-session ids for this connection.
    from deeptutor.services.subagent.sessions import forget_connection

    forget_connection(name)
    return {"status": "disconnected", "name": name}


def _ndjson(obj: dict) -> str:
    return json.dumps(obj, ensure_ascii=False) + "\n"


@router.post("/connections/{name}/message")
async def message_connection(name: str, payload: SubagentMessageRequest):
    """Send a message straight to a connected subagent and stream its run.

    This is the sidebar's "talk to the agent directly" path: it resumes the same
    live session DeepTutor consults (shared via the cross-turn registry, keyed by
    chat session + connection), so the agent keeps full context. Streams the
    native run as newline-delimited JSON, in the same channel shape the chat WS
    uses, so the sidebar transcript renders it identically.
    """
    message = (payload.message or "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="A non-empty 'message' is required.")

    manager = current_kb_manager()
    meta = manager.get_metadata(name)
    if not isinstance(meta, dict) or meta.get("type") != SUBAGENT_KB_TYPE:
        raise HTTPException(status_code=404, detail=f"No connected subagent named {name!r}.")

    from deeptutor.services.subagent.execution import consult_with_session_guard
    from deeptutor.services.subagent.sessions import session_key

    kind = str(meta.get("agent_kind") or "")
    cwd = str(meta.get("cwd") or "")
    partner_id = str(meta.get("partner_id") or "")
    try:
        resolved = resolve_backend_execution(kind, cwd=cwd)
    except SubagentResolutionError as exc:
        _raise_resolution_error(exc)
    backend = resolved.backend
    config = resolved.config
    cwd = resolved.cwd
    provenance = resolved.provenance
    skey = session_key(payload.chat_session_id, name) if payload.chat_session_id else ""

    async def event_stream():
        queue: asyncio.Queue = asyncio.Queue()

        async def on_event(event) -> None:
            await queue.put(("event", event))

        async def run() -> None:
            try:
                res, _execution_key = await consult_with_session_guard(
                    backend,
                    message,
                    on_event=on_event,
                    cwd=cwd,
                    config=config,
                    chat_session_id=payload.chat_session_id,
                    connection=name,
                    session_key_value=skey,
                    partner_id=partner_id or None,
                )
                await queue.put(("done", res))
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("subagent message failed: %s", exc, exc_info=True)
                await queue.put(("fail", str(exc)))

        task = asyncio.create_task(run())
        # The user's own message heads the exchange.
        yield _ndjson(
            {
                "channel": "user_question",
                "text": message,
                "execution_profile": "native",
                "provenance": provenance,
            }
        )
        try:
            while True:
                kind_, item = await queue.get()
                if kind_ == "event":
                    line = {
                        "channel": item.kind,
                        "text": item.text,
                        "execution_profile": "native",
                        "provenance": provenance,
                    }
                    merge_id = (item.meta or {}).get("merge_id")
                    if merge_id:
                        # Namespace away from the chat turn's consult merge ids.
                        line["merge_id"] = f"side:{merge_id}"
                    yield _ndjson(line)
                elif kind_ == "done":
                    yield _ndjson(
                        {
                            "done": True,
                            "success": item.success,
                            "session_id": item.session_id or "",
                            "execution_profile": "native",
                            "provenance": provenance,
                        }
                    )
                    break
                else:  # fail
                    yield _ndjson(
                        {
                            "channel": "error",
                            "text": item,
                            "execution_profile": "native",
                            "provenance": provenance,
                        }
                    )
                    yield _ndjson(
                        {
                            "done": True,
                            "success": False,
                            "execution_profile": "native",
                            "provenance": provenance,
                        }
                    )
                    break
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")


@router.get("/settings")
async def get_settings():
    """Read public turn policy or the admin deployment-backend config."""
    settings = load_subagent_settings()
    if not get_current_user().is_admin:
        return {"consult_budget": settings.consult_budget, "backends": {}}
    return settings.to_dict()


@router.put("/settings", dependencies=[Depends(require_admin)])
async def update_settings(payload: SubagentSettingsPayload):
    """Update the subagent settings (admin-gated; deployment-wide)."""
    merged = load_subagent_settings().to_dict()
    if payload.consult_budget is not None:
        merged["consult_budget"] = payload.consult_budget
    if payload.backends is not None:
        # Merge per backend (and per field) so saving one backend's settings
        # never clobbers the other's or any unsent field.
        backends = dict(merged.get("backends") or {})
        for kind, cfg in payload.backends.items():
            backends[str(kind)] = {**(backends.get(str(kind)) or {}), **(cfg or {})}
        merged["backends"] = backends
    settings = settings_from_dict(merged)
    save_subagent_settings(settings)
    return settings.to_dict()


__all__ = ["router"]
