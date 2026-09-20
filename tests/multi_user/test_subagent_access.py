"""Fail-closed grants and workspace confinement for Connected Agents."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from deeptutor.multi_user.grants import save_grant
from deeptutor.multi_user.subagent_access import allowed_subagent_backends
from deeptutor.services.subagent import access
from deeptutor.services.subagent.config import BackendConfig, SubagentSettings


@pytest.fixture
def grantable_alice(mu_isolated_root, monkeypatch):
    from deeptutor.multi_user import grants

    monkeypatch.setattr(
        grants,
        "get_user_by_id",
        lambda user_id: ("alice", {"role": "user"}) if user_id == "u_alice" else None,
    )
    return "u_alice"


def test_subagent_grants_are_admin_unrestricted_and_user_fail_closed(
    as_user, grantable_alice
) -> None:
    with as_user("u_admin", role="admin"):
        assert allowed_subagent_backends() is None
    with as_user(grantable_alice):
        assert allowed_subagent_backends() == set()

    save_grant(grantable_alice, {"subagent_backends": ["codex", "claude_code"]})
    with as_user(grantable_alice):
        assert allowed_subagent_backends() == {"codex", "claude_code"}


def test_unknown_subagent_backend_cannot_be_saved(grantable_alice) -> None:
    with pytest.raises(ValueError, match="unsupported values"):
        save_grant(grantable_alice, {"subagent_backends": ["invented-runtime"]})


def test_disabled_backend_is_rejected_at_execution_resolution(monkeypatch) -> None:
    backend = SimpleNamespace(kind="codex", local_cli=True)
    monkeypatch.setattr(access, "get_backend", lambda _kind: backend)
    monkeypatch.setattr(access, "subagent_backend_allowed", lambda _kind: True)
    monkeypatch.setattr(
        access,
        "load_subagent_settings",
        lambda: SimpleNamespace(backend=lambda _kind: BackendConfig(enabled=False)),
    )

    with pytest.raises(access.SubagentResolutionError, match="disabled") as excinfo:
        access.resolve_backend_execution("codex", cwd="/workspace")
    assert excinfo.value.forbidden is True


def test_multi_worker_startup_rejects_any_enabled_deployment_backend(monkeypatch) -> None:
    monkeypatch.setattr(access, "list_backend_kinds", lambda: ["codex", "partner"])
    monkeypatch.setattr(
        access,
        "load_subagent_settings",
        lambda: SubagentSettings(backends={"codex": BackendConfig(enabled=True)}),
    )

    with pytest.raises(access.ConnectedAgentConfigurationError, match="enabled: codex"):
        access.assert_connected_agent_worker_configuration(2)


def test_multi_worker_startup_allows_every_deployment_backend_disabled(monkeypatch) -> None:
    monkeypatch.setattr(access, "list_backend_kinds", lambda: ["codex", "partner"])
    monkeypatch.setattr(
        access,
        "load_subagent_settings",
        lambda: SubagentSettings(backends={"codex": BackendConfig(enabled=False)}),
    )

    access.assert_connected_agent_worker_configuration(2)


def test_multi_worker_execution_fails_closed_if_startup_was_bypassed(monkeypatch) -> None:
    backend = SimpleNamespace(kind="partner", local_cli=False)
    monkeypatch.setattr(access, "get_backend", lambda _kind: backend)
    monkeypatch.setattr(
        "deeptutor.services.config.load_system_settings",
        lambda: {"backend_workers": 2},
    )

    with pytest.raises(access.SubagentResolutionError) as excinfo:
        access.resolve_backend_execution("partner", require_workspace=False)
    assert excinfo.value.code == "multi_worker_unsupported"


@pytest.mark.parametrize("kind", ["codex", "partner"])
def test_competing_process_blocks_deployment_and_partner_backends(kind: str, monkeypatch) -> None:
    from deeptutor.services.subagent import process_ownership

    backend = SimpleNamespace(kind=kind, local_cli=kind != "partner")
    monkeypatch.setattr(access, "get_backend", lambda _kind: backend)
    monkeypatch.setattr(
        "deeptutor.services.config.load_system_settings",
        lambda: {"backend_workers": 1},
    )

    def reject_competing_process():
        raise process_ownership.ConnectedAgentProcessOwnershipError("owned elsewhere")

    monkeypatch.setattr(
        process_ownership,
        "ensure_connected_agent_process_ownership",
        reject_competing_process,
    )

    with pytest.raises(access.SubagentResolutionError) as excinfo:
        access.resolve_backend_execution(kind, require_workspace=False)
    assert excinfo.value.code == "multi_worker_unsupported"
    assert excinfo.value.detail == "owned elsewhere"


def test_ungranted_backend_is_rejected_before_workspace_resolution(monkeypatch) -> None:
    backend = SimpleNamespace(kind="codex", local_cli=True)
    monkeypatch.setattr(access, "get_backend", lambda _kind: backend)
    monkeypatch.setattr(access, "subagent_backend_allowed", lambda _kind: False)

    with pytest.raises(access.SubagentResolutionError, match="not granted") as excinfo:
        access.resolve_backend_execution("codex", cwd="")
    assert excinfo.value.code == "backend_not_granted"


def test_local_backend_requires_cwd_inside_selected_workspace(tmp_path: Path, monkeypatch) -> None:
    workspace = tmp_path / "workspace"
    project = workspace / "project"
    outside = tmp_path / "outside"
    project.mkdir(parents=True)
    outside.mkdir()
    binding = SimpleNamespace(workspace_id="ws-test", root=workspace)
    service = SimpleNamespace(
        current_binding=lambda: binding,
        binding_by_id=lambda _workspace_id: binding,
    )
    backend = SimpleNamespace(kind="codex", local_cli=True)
    monkeypatch.setattr(access, "get_backend", lambda _kind: backend)
    monkeypatch.setattr(access, "subagent_backend_allowed", lambda _kind: True)
    monkeypatch.setattr(
        access,
        "load_subagent_settings",
        lambda: SimpleNamespace(backend=lambda _kind: BackendConfig()),
    )
    monkeypatch.setattr(
        "deeptutor.services.workspace.get_content_workspace_service", lambda: service
    )

    with pytest.raises(access.SubagentResolutionError) as empty:
        access.resolve_backend_execution("codex", cwd="")
    assert empty.value.code == "cwd_required"

    with pytest.raises(access.SubagentResolutionError) as escaped:
        access.resolve_backend_execution("codex", cwd=str(outside))
    assert escaped.value.code == "cwd_outside_workspace"

    resolved = access.resolve_backend_execution("codex", cwd=str(project))
    assert resolved.cwd == str(project.resolve())
    assert resolved.provenance == {
        "execution_profile": "native",
        "runtime_owner": "deeptutor",
        "backend_kind": "codex",
        "managed_receipt": False,
    }
