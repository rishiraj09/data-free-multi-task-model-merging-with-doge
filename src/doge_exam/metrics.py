"""Result metrics (experimental protocol section 5.4). Accuracies are percentages (0-100)."""

from __future__ import annotations

from typing import Mapping


def summarize(
    per_task_accuracy: Mapping[str, float],
    expert_accuracy: Mapping[str, float] | None = None,
) -> dict:
    """Mean and worst-task accuracy, plus retention and interference gap when the
    individual-expert accuracy of every task is known."""
    worst_task = min(per_task_accuracy, key=per_task_accuracy.get)
    summary = {
        "mean_accuracy": sum(per_task_accuracy.values()) / len(per_task_accuracy),
        "worst_task": worst_task,
        "worst_task_accuracy": per_task_accuracy[worst_task],
        "normalized_retention": None,
        "mean_normalized_retention": None,
        "interference_gap": None,
    }
    if expert_accuracy and all(t in expert_accuracy for t in per_task_accuracy):
        retention = {t: 100 * a / expert_accuracy[t] for t, a in per_task_accuracy.items()}
        summary["normalized_retention"] = retention
        summary["mean_normalized_retention"] = sum(retention.values()) / len(retention)
        summary["interference_gap"] = {
            t: expert_accuracy[t] - a for t, a in per_task_accuracy.items()
        }
    return summary
