"""Authenticated transport for schema-driven Immersive Reading extensions."""

from __future__ import annotations

import asyncio
import inspect
import logging
import re
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, ValidationError

from deeptutor.multi_user.learning_access import (
    allowed_reading_extensions,
    assert_learning_material,
)
from deeptutor.reading import ReadingStore
from deeptutor.reading.extensions import (
    ReadingContext,
    ReadingExtensionResult,
    get_reading_extension_registry,
)

logger = logging.getLogger(__name__)

router = APIRouter()
ACTION_TIMEOUT_S = 30


class ActionPayload(BaseModel):
    locator: int = Field(ge=1)
    selection: str = Field(default="", max_length=10_000)
    locale: str = Field(default="en", max_length=32)


def _normal(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _verified_selection(candidate: str, unit_text: str) -> str:
    value = _normal(candidate)
    return value if value and value in _normal(unit_text) else ""


def _discard_late_worker_result(worker: asyncio.Future) -> None:
    """Retrieve abandoned worker failures and close unconsumed coroutines."""
    if worker.cancelled():
        return
    try:
        value = worker.result()
        if inspect.iscoroutine(value):
            value.close()
    except Exception:
        logger.exception("Reading extension worker failed after its request ended")


@router.get("/extensions")
async def list_extensions() -> list[dict[str, Any]]:
    allowed = allowed_reading_extensions()
    return [
        extension.manifest.model_dump()
        for extension in get_reading_extension_registry().all()
        if allowed is None or extension.manifest.id in allowed
    ]


@router.post("/materials/{material_id}/extensions/{extension_id}/actions/{action}")
async def run_extension_action(
    material_id: str,
    extension_id: str,
    action: str,
    payload: ActionPayload,
) -> dict[str, Any]:
    try:
        assert_learning_material(material_id)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc

    allowed = allowed_reading_extensions()
    if allowed is not None and extension_id not in allowed:
        raise HTTPException(status_code=403, detail="This reading extension is not allowed.")

    registry = get_reading_extension_registry()
    extension = registry.get(extension_id)
    if extension is None:
        raise HTTPException(status_code=404, detail="Reading extension not found.")
    declared_action = next((row for row in extension.manifest.actions if row.id == action), None)
    if declared_action is None:
        raise HTTPException(status_code=404, detail="Reading extension action not found.")
    store = ReadingStore()
    try:
        unit_text = store.unit_text(material_id, payload.locator)
        position = store.position(material_id)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    selection = _verified_selection(payload.selection, unit_text)
    if "selection" in declared_action.requires and not selection:
        raise HTTPException(status_code=400, detail="Select text from the visible unit first.")
    try:
        context = ReadingContext(
            material_id=material_id,
            locator=payload.locator,
            source_anchor=(position.source_anchor if position.locator == payload.locator else ""),
            locale=payload.locale,
            selection=selection,
            visible_text=unit_text,
        )
    except ValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail="This reading unit is too large for the extension protocol.",
        ) from exc
    if not registry.begin_action(extension_id):
        raise HTTPException(
            status_code=503,
            detail={
                "message": "This reading action is temporarily unavailable.",
                "recoverable": True,
            },
        )
    # Only a still-running worker needs the circuit kept open (#1448).
    # Async cancellation finishes before the reservation is released, including
    # sync handlers that return an awaitable after their worker has finished.
    worker: asyncio.Future | None = None
    try:
        async with asyncio.timeout(ACTION_TIMEOUT_S):
            handler = extension.run_action
            if inspect.iscoroutinefunction(handler):
                value = await handler(action, context)
            else:
                worker = asyncio.get_running_loop().run_in_executor(
                    registry.executor_for(extension_id), handler, action, context
                )
                value = await asyncio.shield(worker)
            if inspect.isawaitable(value):
                value = await value
        result = (
            value
            if isinstance(value, ReadingExtensionResult)
            else ReadingExtensionResult.model_validate(value)
        )
        if result.type not in extension.manifest.result_types:
            raise ValueError(f"Extension returned undeclared result type {result.type!r}.")
        return result.model_dump()
    except TimeoutError as exc:
        logger.warning("Reading extension %s action %s timed out", extension_id, action)
        raise HTTPException(
            status_code=503,
            detail={
                "message": "This reading action is temporarily unavailable.",
                "recoverable": True,
            },
        ) from exc
    except Exception as exc:
        logger.exception("Reading extension %s action %s failed", extension_id, action)
        raise HTTPException(
            status_code=503,
            detail={
                "message": "This reading action is temporarily unavailable.",
                "recoverable": True,
            },
        ) from exc
    finally:
        if worker is not None and not worker.done():
            registry.mark_timed_out(extension_id)
            worker.add_done_callback(_discard_late_worker_result)
        registry.finish_action(extension_id)


__all__ = ["router"]
