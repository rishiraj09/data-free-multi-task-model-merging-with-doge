"""Portable device selection. CUDA is canonical; MPS/CPU are explicit fallbacks."""

from __future__ import annotations

import os

import torch

VALID_DEVICES = ("cuda", "mps", "cpu")


def select_device(preferred: str | None = None) -> torch.device:
    """Return the requested device, or the best available one (cuda > mps > cpu).

    The DOGE_DEVICE environment variable overrides auto-detection. Requesting an
    unavailable device raises instead of silently falling back.
    """
    preferred = preferred or os.environ.get("DOGE_DEVICE")
    if preferred:
        if preferred not in VALID_DEVICES:
            raise ValueError(f"Unknown device {preferred!r}; expected one of {VALID_DEVICES}")
        if not is_available(preferred):
            raise RuntimeError(f"Requested device {preferred!r} is not available")
        return torch.device(preferred)
    for name in VALID_DEVICES:
        if is_available(name):
            return torch.device(name)
    raise RuntimeError("No device available")  # unreachable: cpu is always available


def is_available(name: str) -> bool:
    if name == "cuda":
        return torch.cuda.is_available()
    if name == "mps":
        return torch.backends.mps.is_available()
    return name == "cpu"


def is_canonical(device: torch.device) -> bool:
    """Only CUDA runs count as canonical results (see experimental protocol, section 9)."""
    return device.type == "cuda"
