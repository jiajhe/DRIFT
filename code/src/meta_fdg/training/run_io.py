from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import traceback
from typing import Any

import torch


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(_jsonable(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _git_output(root: Path, *arguments: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *arguments], cwd=root, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


@dataclass
class RunDirectory:
    root: Path
    repo_root: Path

    @classmethod
    def create(cls, repo_root: Path, config: dict, *, allow_existing: bool = False) -> "RunDirectory":
        tier = str(config["tier"])
        job_id = str(config["job_id"])
        root = repo_root / "runs" / tier / job_id
        if root.exists() and not allow_existing:
            raise FileExistsError(f"run already exists: {root}")
        for directory in (root, root / "logs", root / "checkpoints"):
            directory.mkdir(parents=True, exist_ok=True)
        instance = cls(root=root, repo_root=repo_root)
        atomic_json(root / "resolved_config.json", config)
        atomic_json(root / "environment.json", instance.environment())
        instance.set_status("pending")
        return instance

    def environment(self) -> dict:
        git_status = _git_output(self.repo_root, "status", "--short")
        return {
            "captured_at_utc": _now(),
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_devices": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
            "git_revision": _git_output(self.repo_root, "rev-parse", "HEAD"),
            "git_dirty": bool(git_status),
            "git_status": git_status or "",
        }

    def set_status(self, status: str, **extra: Any) -> None:
        payload = {"status": status, "updated_at_utc": _now(), **extra}
        atomic_json(self.root / "status.json", payload)

    def fail(self, error: BaseException) -> None:
        self.set_status(
            "failed",
            error_type=type(error).__name__,
            error=str(error),
            traceback=traceback.format_exc(),
        )
