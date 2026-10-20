"""Configurable storage roots.

Resolution order, later wins:
1. configs/paths.yaml (checked-in defaults)
2. configs/paths.local.yaml (git-ignored, per machine)
3. DOGE_<KEY> environment variables, e.g. DOGE_DATA_ROOT
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Mapping

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "paths.yaml"
LOCAL_CONFIG = PROJECT_ROOT / "configs" / "paths.local.yaml"


@dataclass(frozen=True)
class ProjectPaths:
    data_root: Path
    checkpoint_root: Path
    cache_root: Path
    results_root: Path
    figures_root: Path

    @property
    def raw_results(self) -> Path:
        return self.results_root / "raw"

    @property
    def processed_results(self) -> Path:
        return self.results_root / "processed"

    @property
    def manifests(self) -> Path:
        return self.results_root / "manifests"

    @property
    def hf_home(self) -> Path:
        return self.cache_root / "huggingface"

    def as_dict(self) -> dict[str, str]:
        return {f.name: str(getattr(self, f.name)) for f in fields(self)}

    def ensure(self) -> None:
        for path in (
            self.data_root,
            self.checkpoint_root,
            self.cache_root,
            self.raw_results,
            self.processed_results,
            self.manifests,
            self.figures_root,
        ):
            path.mkdir(parents=True, exist_ok=True)


def _read_yaml(path: Path) -> dict:
    if not path.is_file():
        return {}
    with path.open() as f:
        return yaml.safe_load(f) or {}


def _resolve(value: str | os.PathLike, base: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def load_paths(
    config_path: Path | None = None,
    local_config_path: Path | None = None,
    env: Mapping[str, str] | None = None,
    base: Path = PROJECT_ROOT,
) -> ProjectPaths:
    env = os.environ if env is None else env
    values = _read_yaml(config_path or DEFAULT_CONFIG)
    values.update(_read_yaml(local_config_path or LOCAL_CONFIG))

    resolved = {}
    for f in fields(ProjectPaths):
        raw = env.get(f"DOGE_{f.name.upper()}", values.get(f.name))
        if raw is None:
            raise KeyError(f"No value configured for {f.name}")
        resolved[f.name] = _resolve(raw, base)
    return ProjectPaths(**resolved)


def configure_hf_cache(paths: ProjectPaths, env: dict | None = None) -> None:
    """Point Hugging Face caches at cache_root unless the user already set HF_HOME."""
    env = os.environ if env is None else env
    env.setdefault("HF_HOME", str(paths.hf_home))
