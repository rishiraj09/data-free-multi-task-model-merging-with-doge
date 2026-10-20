"""Checkpoint integrity helpers (CKPT-03, CKPT-04, CKPT-05).

Network access lives in scripts/verify_inventory.py; everything here works on local
files and state dicts so it can be unit-tested with synthetic tensors.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Mapping

import torch

# Experimental protocol section 6.3: float32 absolute and relative tolerance of 1e-5.
ATOL = 1e-5
RTOL = 1e-5


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        while chunk := f.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def compare_architecture(
    reference: Mapping[str, torch.Tensor], other: Mapping[str, torch.Tensor]
) -> dict:
    """CKPT-03: same parameter keys and shapes as the reference (pretrained) state dict."""
    ref_keys, other_keys = set(reference), set(other)
    shape_mismatches = sorted(
        k for k in ref_keys & other_keys if tuple(reference[k].shape) != tuple(other[k].shape)
    )
    return {
        "missing_keys": sorted(ref_keys - other_keys),
        "extra_keys": sorted(other_keys - ref_keys),
        "shape_mismatches": shape_mismatches,
        "passed": ref_keys == other_keys and not shape_mismatches,
    }


def task_vector_roundtrip(
    pretrained: Mapping[str, torch.Tensor],
    finetuned: Mapping[str, torch.Tensor],
    atol: float = ATOL,
    rtol: float = RTOL,
) -> dict:
    """CKPT-05: theta_0 + (theta_i - theta_0) reconstructs theta_i within tolerance.

    Also reports the task-vector norm, so a checkpoint identical to the pretrained
    model (an empty task vector) is caught instead of passing silently.
    """
    max_abs_error = 0.0
    sq_norm = 0.0
    all_close = True
    for key, theta_0 in pretrained.items():
        theta_i = finetuned[key]
        if not theta_0.is_floating_point():
            continue
        tau = theta_i - theta_0
        rebuilt = theta_0 + tau
        max_abs_error = max(max_abs_error, (rebuilt - theta_i).abs().max().item())
        all_close &= torch.allclose(rebuilt, theta_i, atol=atol, rtol=rtol)
        sq_norm += tau.double().pow(2).sum().item()
    tau_norm = sq_norm**0.5
    return {
        "max_abs_error": max_abs_error,
        "atol": atol,
        "rtol": rtol,
        "task_vector_l2_norm": tau_norm,
        "passed": bool(all_close) and tau_norm > 0,
    }
