from __future__ import annotations

import pytest

from deeptutor.runtime.coordination import (
    CoordinationSettings,
    RuntimeConfigurationError,
)


def test_multi_worker_requires_redis() -> None:
    with pytest.raises(RuntimeConfigurationError, match="requires"):
        CoordinationSettings.from_runtime_settings(
            {"backend_workers": 4},
            {"turn_coordination": {"backend": "memory"}},
        )


def test_redis_requires_url() -> None:
    with pytest.raises(RuntimeConfigurationError, match="redis_url"):
        CoordinationSettings.from_runtime_settings(
            {"backend_workers": 2},
            {"turn_coordination": {"backend": "redis", "redis_url": ""}},
        )


def test_runtime_report_never_exposes_redis_credentials() -> None:
    settings = CoordinationSettings.from_runtime_settings(
        {"backend_workers": 4},
        {
            "turn_coordination": {
                "backend": "redis",
                "redis_url": "redis://:secret@redis:6379/0",
            }
        },
    )

    report = settings.runtime_report()
    assert report["redis_configured"] is True
    assert "secret" not in repr(report)
    assert "redis_url" not in report


def test_application_container_rejects_enabled_connected_agents_with_multiple_workers(
    monkeypatch,
) -> None:
    from deeptutor.app import container as container_module
    from deeptutor.services.subagent import access as subagent_access

    monkeypatch.setattr(
        container_module,
        "load_system_settings",
        lambda: {"backend_workers": 2},
    )
    monkeypatch.setattr(
        container_module,
        "load_integrations_settings",
        lambda: {"turn_coordination": {"backend": "redis", "redis_url": "redis://test"}},
    )
    monkeypatch.setattr(
        subagent_access,
        "enabled_deployment_backend_kinds",
        lambda: {"codex"},
    )

    with pytest.raises(RuntimeConfigurationError, match="Connected Agent backend"):
        container_module.ApplicationContainer.build()
