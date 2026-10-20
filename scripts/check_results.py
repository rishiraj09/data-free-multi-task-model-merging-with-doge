"""Check every newest complete merge record in results/raw (INT-04 and the merge checks).

For each record: the experimental protocol section 7.2 fields are present; merged weights are finite; no data
was accessed during merging; the save/reload check passed; test data was loaded only after
the merge; DOGE records pass the probe checks (objective finite, ||Pg||/||g|| <= 1e-3).
Also checks that the expected experiment x method runs exist.

Usage: python scripts/check_results.py [EXPERIMENT ...]   (exit status 1 on any failure)
"""

from __future__ import annotations

import sys

import yaml

from doge_exam.paths import PROJECT_ROOT, load_paths

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from make_table import latest_records  # noqa: E402

# Experimental protocol section 7.2 field -> key in the result records
REQUIRED = {
    "run_id": "run_id", "timestamp": "timestamp", "git_commit": "provenance", "method": "method",
    "task_set": "task_set", "task_count": "task_count", "seed": "seed", "base_model": "base_model",
    "checkpoint identifiers": "checkpoints", "configuration": "method_config", "device": "device",
    "precision": "precision", "software versions": "software", "per_task_accuracy": "per_task_accuracy",
    "mean_accuracy": "mean_accuracy", "worst_task_accuracy": "worst_task_accuracy",
    "normalized_retention": "normalized_retention", "merge_time_seconds": "merge_time_seconds",
    "evaluation_time_seconds": "evaluation_time_seconds", "peak_memory_mb": "peak_memory_mb",
    "status": "status", "notes": "notes",
}
PROJECTION_TOLERANCE = 1e-3  # float32 tolerance rationale: docs/PAPER_TO_CODE_MAP.md (DOGE-03)


def check(record: dict) -> list[str]:
    problems = [f"missing {name}" for name, key in REQUIRED.items() if record.get(key) is None]
    if not record.get("merged_weights_finite"):
        problems.append("merged weights not finite")
    if record.get("data_access_during_merge", {}).get("data_accessed") is not False:
        problems.append("data accessed during merge (or not recorded)")
    if not record.get("reload_check", {}).get("passed"):
        problems.append("reload check failed or missing")
    if not record.get("merge_finished_at", "z") < record.get("test_data_first_loaded_at", ""):
        problems.append("test data loaded before the merge finished")
    if record["method"] == "doge-ta":
        probe = record.get("doge_probe", {})
        if not (probe.get("objective_all_finite") and probe.get("delta_finite")):
            problems.append("DOGE objective or delta not finite")
        if not probe.get("max_projected_grad_in_subspace_ratio", 1) <= PROJECTION_TOLERANCE:
            problems.append("DOGE projection residual above tolerance")
        if probe.get("iterations_per_layer") != [400] or probe.get("layers") != 72:
            problems.append("DOGE did not run 400 iterations on 72 layers")
    return problems


def main() -> int:
    paths = load_paths()
    experiments = yaml.safe_load((PROJECT_ROOT / "configs/experiments.yaml").read_text())["experiments"]
    methods = yaml.safe_load((PROJECT_ROOT / "configs/canonical_methods.yaml").read_text())["methods"]
    wanted = sys.argv[1:] or list(experiments)
    latest = {rid: rec for rid, (_, rec) in latest_records(paths.raw_results).items() if rec["kind"] == "merge"}
    failures = 0
    for experiment in wanted:
        for method in methods:
            found = [r for r in latest.values() if r["experiment"] == experiment and r["method"] == method]
            if not found:
                print(f"{experiment} {method}: MISSING")
                failures += 1
                continue
            problems = check(found[0])
            failures += bool(problems)
            print(f"{experiment} {method}: {'OK' if not problems else 'FAIL ' + '; '.join(problems)}")
    print(f"{failures} problem(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
