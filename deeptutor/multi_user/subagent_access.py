"""Per-account authorization for deployment-owned Connected Agent backends."""

from __future__ import annotations

from .context import get_current_user
from .grants import load_grant


def allowed_subagent_backends() -> set[str] | None:
    """Return allowed backend ids; ``None`` is administrator-unrestricted.

    Local CLIs inherit the server account's credentials and configured remote
    gateways inherit deployment credentials.  Unlike ordinary optional tools,
    both are therefore deny-by-default for every non-admin account.
    """

    user = get_current_user()
    if user.is_admin:
        return None
    value = load_grant(user.id).get("subagent_backends")
    if value is None:
        return set()
    return {str(kind).strip() for kind in value if str(kind).strip()}


def subagent_backend_allowed(kind: str) -> bool:
    """Whether the current account may use one deployment backend id."""

    allowed = allowed_subagent_backends()
    return allowed is None or str(kind or "").strip() in allowed


__all__ = ["allowed_subagent_backends", "subagent_backend_allowed"]
