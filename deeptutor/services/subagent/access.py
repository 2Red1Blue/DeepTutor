"""Fail-closed resolution for Connected Agent execution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from deeptutor.multi_user.subagent_access import subagent_backend_allowed
from deeptutor.services.subagent.base import SubagentBackend
from deeptutor.services.subagent.config import BackendConfig, load_subagent_settings
from deeptutor.services.subagent.partner import PARTNER_BACKEND_KIND
from deeptutor.services.subagent.registry import get_backend, list_backend_kinds


class ConnectedAgentConfigurationError(ValueError):
    """The deployment cannot safely execute Connected Agent backends."""


class SubagentResolutionError(ValueError):
    """A backend cannot execute under the current account and workspace."""

    def __init__(self, code: str, detail: str, *, forbidden: bool = False) -> None:
        super().__init__(detail)
        self.code = code
        self.detail = detail
        self.forbidden = forbidden


@dataclass(frozen=True, slots=True)
class ResolvedSubagent:
    """Authorized backend, effective settings, and confined working directory."""

    backend: SubagentBackend
    config: BackendConfig
    cwd: str

    @property
    def provenance(self) -> dict[str, object]:
        """Stable marker distinguishing native execution from managed runtimes."""

        return native_execution_provenance(self.backend.kind)


def native_execution_provenance(kind: str) -> dict[str, object]:
    """Return the non-authoritative execution provenance exposed to clients."""

    return {
        "execution_profile": "native",
        "runtime_owner": "deeptutor",
        "backend_kind": str(kind or "").strip(),
        "managed_receipt": False,
    }


def backend_available_to_current_user(kind: str) -> bool:
    """Whether discovery/configuration may expose one deployment backend."""

    normalized = str(kind or "").strip()
    if not normalized or normalized == PARTNER_BACKEND_KIND:
        return normalized == PARTNER_BACKEND_KIND
    if not subagent_backend_allowed(normalized):
        return False
    return load_subagent_settings().backend(normalized).enabled


def executable_backend_kinds() -> set[str]:
    """Return backend ids the current account may probe or execute."""

    candidates = {
        kind
        for kind in list_backend_kinds()
        if kind != PARTNER_BACKEND_KIND and backend_available_to_current_user(kind)
    }
    return candidates


def enabled_deployment_backend_kinds() -> set[str]:
    """Return enabled deployment backends governed by ``subagent.json``.

    Partner execution has its own authorization and no deployment-backend
    toggle, so it is fenced at invocation time instead of participating in the
    startup configuration check.
    """

    settings = load_subagent_settings()
    return {
        kind
        for kind in list_backend_kinds()
        if kind != PARTNER_BACKEND_KIND and settings.backend(kind).enabled
    }


def assert_connected_agent_worker_configuration(backend_workers: int) -> None:
    """Reject multi-worker startup while a native backend is enabled.

    The session registry is process-local and has no distributed lease. Redis
    turn coordination therefore does not make Connected Agent session resume
    safe across API workers.
    """

    workers = max(1, int(backend_workers))
    if workers <= 1:
        return
    enabled = sorted(enabled_deployment_backend_kinds())
    if not enabled:
        return
    raise ConnectedAgentConfigurationError(
        "backend_workers > 1 requires every Connected Agent backend to be disabled "
        "until the backend-session registry has a distributed lease; enabled: " + ", ".join(enabled)
    )


def _assert_single_worker_execution() -> None:
    """Fence every native invocation even when startup validation was bypassed."""

    from deeptutor.services.config import load_system_settings

    workers = max(1, int(load_system_settings().get("backend_workers") or 1))
    if workers > 1:
        raise SubagentResolutionError(
            "multi_worker_unsupported",
            "Connected Agents are unavailable when backend_workers > 1 until their "
            "backend-session registry has a distributed lease.",
        )


def resolve_backend_execution(
    kind: str, *, cwd: str = "", require_workspace: bool = True
) -> ResolvedSubagent:
    """Authorize and resolve one native Connected Agent invocation.

    Partners use their separate per-partner grant and have no host working
    directory.  Every deployment backend is re-authorized at invocation time,
    so revoking a grant or disabling a backend also closes existing pointers.
    Local CLIs additionally require an explicit directory contained by the
    caller's currently selected content workspace.
    """

    normalized = str(kind or "").strip()
    backend = get_backend(normalized)
    if backend is None:
        raise SubagentResolutionError("unknown_backend", f"Unknown agent kind: {normalized!r}")
    _assert_single_worker_execution()
    if normalized == PARTNER_BACKEND_KIND:
        return ResolvedSubagent(backend=backend, config=BackendConfig(), cwd="")
    if not subagent_backend_allowed(normalized):
        raise SubagentResolutionError(
            "backend_not_granted",
            f"Connected Agent backend {normalized!r} is not granted to this account.",
            forbidden=True,
        )
    config = load_subagent_settings().backend(normalized)
    if not config.enabled:
        raise SubagentResolutionError(
            "backend_disabled",
            f"Connected Agent backend {normalized!r} is disabled by the operator.",
            forbidden=True,
        )
    resolved_cwd = (
        _resolve_owner_workspace_cwd(cwd)
        if getattr(backend, "local_cli", True) and require_workspace
        else ""
    )
    return ResolvedSubagent(backend=backend, config=config, cwd=resolved_cwd)


def _resolve_owner_workspace_cwd(raw_cwd: str) -> str:
    value = str(raw_cwd or "").strip()
    if not value:
        raise SubagentResolutionError(
            "cwd_required",
            "Local Connected Agents require an explicit working directory.",
        )
    path = Path(value).expanduser()
    if not path.exists() or not path.is_dir():
        raise SubagentResolutionError(
            "cwd_invalid", f"Connected Agent working directory is not a directory: {path}"
        )
    resolved = path.resolve()

    from deeptutor.services.workspace import get_content_workspace_service

    service = get_content_workspace_service()
    binding = service.current_binding()
    try:
        binding = service.binding_by_id(binding.workspace_id)
        resolved.relative_to(binding.root.resolve())
    except (ValueError, OSError) as exc:
        raise SubagentResolutionError(
            "cwd_outside_workspace",
            "The working directory must be inside the account's selected workspace.",
            forbidden=True,
        ) from exc
    return str(resolved)


__all__ = [
    "ConnectedAgentConfigurationError",
    "ResolvedSubagent",
    "SubagentResolutionError",
    "assert_connected_agent_worker_configuration",
    "backend_available_to_current_user",
    "enabled_deployment_backend_kinds",
    "executable_backend_kinds",
    "native_execution_provenance",
    "resolve_backend_execution",
]
