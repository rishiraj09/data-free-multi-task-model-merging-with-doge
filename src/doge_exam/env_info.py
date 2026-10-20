"""Collect reproducibility metadata about the software and hardware environment."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

from doge_exam.device import is_canonical, select_device
from doge_exam.paths import PROJECT_ROOT

TRACKED_PACKAGES = (
    "torch",
    "torchvision",
    "transformers",
    "datasets",
    "huggingface_hub",
    "lightning",
    "hydra-core",
    "peft",
    "numpy",
    "scipy",
    "fusion_bench",
)

DOGE_REPO = PROJECT_ROOT / "third_party" / "DOGE"


def package_versions() -> dict[str, str | None]:
    versions = {}
    for name in TRACKED_PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def git_commit(repo: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip()


def device_info() -> dict:
    device = select_device()
    info = {
        "selected_device": str(device),
        "canonical": is_canonical(device),
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        "gpus": [],
    }
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        info["gpus"].append({"name": props.name, "total_memory_mb": props.total_memory // 2**20})
    return info


def collect_environment() -> dict:
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": sys.version.split()[0],
        "python_executable": sys.executable,
        "packages": package_versions(),
        "project_git_commit": git_commit(PROJECT_ROOT),
        "doge_repo_commit": git_commit(DOGE_REPO),
        "device": device_info(),
    }


def write_environment_record(out_dir: Path) -> Path:
    record = collect_environment()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"env__{record['device']['selected_device']}__{stamp}.json"
    path.write_text(json.dumps(record, indent=2) + "\n")
    return path
