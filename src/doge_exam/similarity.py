"""Task-vector cosine similarity, definition `encw-cos`.

tau_i = theta_i - theta_0 over the CLIPVisionModel state-dict keys that contain "encoder" and
"weight" but not "layer_norm" (the parameters DOGE operates on), flattened in sorted key order;
cosine computed in float64.
"""

from __future__ import annotations

import itertools
from typing import Mapping

import torch

DEFINITION_ID = "encw-cos"


def selected_keys(state_dict: Mapping[str, torch.Tensor]) -> list[str]:
    return sorted(k for k in state_dict if "encoder" in k and "weight" in k and "layer_norm" not in k)


def task_vector(pretrained: Mapping[str, torch.Tensor], finetuned: Mapping[str, torch.Tensor]) -> torch.Tensor:
    keys = selected_keys(pretrained)
    return torch.cat([(finetuned[k].double() - pretrained[k].double()).flatten() for k in keys])


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return (torch.dot(a, b) / (a.norm() * b.norm())).item()


def pairwise_cosine(vectors: Mapping[str, torch.Tensor]) -> dict[str, float]:
    """Keys are 'taskA-taskB' in the insertion order of `vectors`."""
    return {f"{a}-{b}": cosine(vectors[a], vectors[b]) for a, b in itertools.combinations(vectors, 2)}
