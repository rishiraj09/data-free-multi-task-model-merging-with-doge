"""Observation probes for merge runs (DOGE-02, DOGE-03, DOGE-05, MATH-03/04/07).

The probes wrap the official objects at run time; the official DOGE code is not modified.

- DataAccessProbe (any method): counts every dataset load, dataset item read and DataLoader
  iteration while `method.run` executes. Data-free merging must leave all counts at zero.
- DogeProbe (DOGE TA): records the objective of every iteration (kept on the GPU, no sync),
  and at iterations 1, 200 and 400 of every layer checks that the gradient passed to Adam
  has no component in the shared subspace (||P g|| / ||g||), that P is a projection
  (||P P - P||), and how much of the accumulated Delta lies in the subspace.
"""

from __future__ import annotations

import math
from collections import defaultdict

import torch
import torch.optim.optimizer as optim_mod
from torch.utils.data import DataLoader

CHECK_STEPS = (1, 200, 400)


class DataAccessProbe:
    def __init__(self):
        self.counts = defaultdict(int)
        self._patches = []

    def _wrap(self, owner, name: str, label: str):
        original = getattr(owner, name)
        counts = self.counts

        def wrapper(*args, **kwargs):
            counts[label] += 1
            return original(*args, **kwargs)

        setattr(owner, name, wrapper)
        self._patches.append((owner, name, original))

    def __enter__(self):
        import datasets
        import datasets.arrow_dataset
        from fusion_bench.modelpool import BaseModelPool

        self._wrap(datasets, "load_dataset", "datasets.load_dataset")
        self._wrap(datasets, "load_from_disk", "datasets.load_from_disk")
        self._wrap(datasets.arrow_dataset.Dataset, "__getitem__", "datasets.Dataset.__getitem__")
        self._wrap(DataLoader, "__iter__", "DataLoader.__iter__")
        for name in ("load_train_dataset", "load_val_dataset", "load_test_dataset"):
            self._wrap(BaseModelPool, name, f"modelpool.{name}")
        return self

    def __exit__(self, *exc):
        for owner, name, original in reversed(self._patches):
            setattr(owner, name, original)
        return False

    def summary(self) -> dict:
        return {"counts": dict(self.counts), "data_accessed": any(self.counts.values())}


class DogeProbe:
    def __init__(self, algorithm):
        self.algorithm = algorithm
        self.losses = defaultdict(list)
        self.checks = []
        self.current, self.step = None, 0
        self._names = None
        self._hook = None
        self._original_loss = algorithm.taskvector_loss

    def __enter__(self):
        algo = self.algorithm

        def loss_wrapper(layer_vectors, layer_delta, layer_lamdas):
            if self._names is None:
                self._names = {id(p): k for k, p in algo.delta.items()}
            name = self._names[id(layer_delta)]
            if name != self.current:
                self.current, self.step = name, 0
            self.step += 1
            loss = self._original_loss(layer_vectors, layer_delta, layer_lamdas)
            self.losses[name].append(loss.detach())
            return loss

        def step_hook(optimizer, args, kwargs):
            if self.step not in CHECK_STEPS:
                return
            delta = algo.delta[self.current]
            grad = delta.grad
            P = algo.projection[self.current]
            entry = {
                "layer": self.current,
                "iteration": self.step,
                "grad_finite": bool(torch.isfinite(grad).all()),
                "delta_finite": bool(torch.isfinite(delta).all()),
                "projected_grad_in_subspace_ratio": ((P @ grad).norm() / grad.norm().clamp_min(1e-30)).item(),
            }
            if self.step == CHECK_STEPS[0]:
                entry["projection_idempotence_error"] = (P @ P - P).abs().max().item()
                entry["projection_rank"] = round(torch.trace(P).item())
            else:
                d_norm = delta.detach().norm()
                entry["delta_in_subspace_ratio"] = (
                    ((P @ delta.detach()).norm() / d_norm).item() if d_norm > 0 else 0.0
                )
            self.checks.append(entry)

        algo.taskvector_loss = loss_wrapper
        self._hook = optim_mod.register_optimizer_step_pre_hook(step_hook)
        return self

    def __exit__(self, *exc):
        self.algorithm.taskvector_loss = self._original_loss
        if self._hook is not None:
            self._hook.remove()
        return False

    def summary(self) -> dict:
        per_layer = {}
        all_finite = True
        for name, values in self.losses.items():
            v = torch.stack(values).float().cpu()
            finite = bool(torch.isfinite(v).all())
            all_finite &= finite
            per_layer[name] = {
                "iterations": len(values),
                "first": v[0].item(),
                "last": v[-1].item(),
                "min": v.min().item(),
                "max": v.max().item(),
                "finite": finite,
            }
        delta_finite = all(bool(torch.isfinite(d).all()) for d in self.algorithm.delta.values())
        lamdas_finite = all(
            math.isfinite(float(x)) for lam in self.algorithm.lamdas for x in lam.values()
        )
        ratios = [c["projected_grad_in_subspace_ratio"] for c in self.checks]
        idem = [c["projection_idempotence_error"] for c in self.checks if "projection_idempotence_error" in c]
        sub = [c["delta_in_subspace_ratio"] for c in self.checks if "delta_in_subspace_ratio" in c]
        return {
            "layers": len(per_layer),
            "iterations_per_layer": sorted({p["iterations"] for p in per_layer.values()}),
            "objective_all_finite": all_finite,
            "layers_with_lower_final_objective": sum(p["last"] < p["first"] for p in per_layer.values()),
            "delta_finite": delta_finite,
            "lamdas_finite": lamdas_finite,
            "grads_finite_at_checks": all(c["grad_finite"] for c in self.checks),
            "max_projected_grad_in_subspace_ratio": max(ratios) if ratios else None,
            "max_projection_idempotence_error": max(idem) if idem else None,
            "projection_ranks": sorted({c["projection_rank"] for c in self.checks if "projection_rank" in c}),
            "max_delta_in_subspace_ratio": max(sub) if sub else None,
            "per_layer_objective": per_layer,
            "checks": self.checks,
        }
