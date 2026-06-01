from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


def _run_command(args: list[str]) -> str | None:
    try:
        result = subprocess.run(
            args,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None

    return result.stdout.strip() or None


def get_git_commit() -> str | None:
    return _run_command(["git", "rev-parse", "HEAD"])


def get_git_branch() -> str | None:
    return _run_command(["git", "branch", "--show-current"])


def get_git_status_short() -> str | None:
    return _run_command(["git", "status", "--short"])


def build_run_manifest(
    *,
    config: dict[str, Any],
    command_line: list[str],
    device: torch.device,
    dry_run: bool = False,
) -> dict[str, Any]:
    cuda_available: bool | str

    if dry_run:
        cuda_available = "no consultado en dry-run"
        cuda_device_name = "no consultado en dry-run"
        cuda_device_count = "no consultado en dry-run"
    else:
        cuda_available = torch.cuda.is_available()
        cuda_device_count = torch.cuda.device_count() if cuda_available else 0
        cuda_device_name = torch.cuda.get_device_name(0) if cuda_available else None

    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command_line": command_line,
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
        },
        "torch": {
            "version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": cuda_available,
            "cuda_device_count": cuda_device_count,
            "cuda_device_name": cuda_device_name,
            "device_selected": str(device),
        },
        "git": {
            "commit": get_git_commit(),
            "branch": get_git_branch(),
            "status_short": get_git_status_short(),
        },
        "config": config,
    }


def write_run_manifest(
    *,
    path: str | Path,
    manifest: dict[str, Any],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, ensure_ascii=False)
