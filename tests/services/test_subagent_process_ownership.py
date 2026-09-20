"""Cross-process acceptance for Connected Agent native execution ownership."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import time

_HOLDER = """
from pathlib import Path
import sys
import time
from deeptutor.services.subagent.access import resolve_backend_execution

resolve_backend_execution("partner", require_workspace=False)
Path(sys.argv[1]).write_text("ready", encoding="utf-8")
time.sleep(30)
"""

_CONTENDER = """
from deeptutor.services.subagent.access import SubagentResolutionError, resolve_backend_execution

try:
    resolve_backend_execution("partner", require_workspace=False)
except SubagentResolutionError as exc:
    print(exc.code)
    print(exc.detail)
else:
    raise SystemExit("competing process unexpectedly acquired Connected Agent execution")
"""

_SUCCESSOR = """
from deeptutor.services.subagent.access import resolve_backend_execution

resolved = resolve_backend_execution("partner", require_workspace=False)
print(resolved.backend.kind)
"""


def _wait_for(path: Path, process: subprocess.Popen[str]) -> None:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if path.exists():
            return
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            raise AssertionError(
                f"ownership holder exited {process.returncode}: {stdout}\n{stderr}"
            )
        time.sleep(0.02)
    raise AssertionError("ownership holder did not become ready")


def test_real_second_process_cannot_bypass_worker_setting(tmp_path: Path) -> None:
    """A launcher claiming one worker cannot grant two processes authority."""

    runtime_home = tmp_path / "runtime"
    ready = tmp_path / "ready"
    env = os.environ.copy()
    env["DEEPTUTOR_HOME"] = str(runtime_home)
    holder = subprocess.Popen(
        [sys.executable, "-c", _HOLDER, str(ready)],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        _wait_for(ready, holder)
        contender = subprocess.run(
            [sys.executable, "-c", _CONTENDER],
            cwd=Path(__file__).resolve().parents[2],
            env=env,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        assert contender.returncode == 0, contender.stderr
        assert contender.stdout.splitlines()[0] == "multi_worker_unsupported"
        assert "already owned by another local DeepTutor process" in contender.stdout
    finally:
        holder.terminate()
        try:
            holder.wait(timeout=10)
        except subprocess.TimeoutExpired:
            holder.kill()
            holder.wait(timeout=10)

    successor = subprocess.run(
        [sys.executable, "-c", _SUCCESSOR],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert successor.returncode == 0, successor.stderr
    assert successor.stdout.strip() == "partner"


def test_process_ownership_module_keeps_platform_locks_lazy() -> None:
    from deeptutor.services.subagent import process_ownership

    assert "fcntl" not in process_ownership.__dict__
    assert "msvcrt" not in process_ownership.__dict__
