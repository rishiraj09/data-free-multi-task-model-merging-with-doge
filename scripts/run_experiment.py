"""Run reference evaluations, merges, or the BASE-03 determinism check.

Examples (from the project root, project venv active):
  python scripts/run_experiment.py references [--fast-dev-run]
  python scripts/run_experiment.py merge P01 task-arithmetic [--fast-dev-run]
  python scripts/run_experiment.py determinism P01 task-arithmetic

Complete runs go to results/raw/<run_id>__<timestamp>.json; --fast-dev-run (one batch per
task) goes to results/smoke/. Settings come from configs/canonical_methods.yaml,
configs/experiments.yaml and configs/pins.yaml.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

from doge_exam.paths import configure_hf_cache, load_paths

PATHS = load_paths()
PATHS.ensure()
configure_hf_cache(PATHS)  # before huggingface_hub is imported

from doge_exam import runner  # noqa: E402


def determinism(experiment: str, method: str) -> None:
    """BASE-03: merging twice with the same inputs must give identical weights."""
    fabric = runner.make_fabric()
    hashes = [runner.merge(experiment, method, fabric)[1]["merged_weights_sha256"] for _ in range(2)]
    record = {
        "kind": "determinism",
        "test_id": "BASE-03",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_id": runner.run_id_for(experiment, method),
        "merged_weights_sha256": hashes,
        "identical": hashes[0] == hashes[1],
        "provenance": runner.provenance(),
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = PATHS.manifests / f"determinism__{record['run_id']}__{stamp}.json"
    out.write_text(json.dumps(record, indent=2))
    print(f"BASE-03 identical={record['identical']}\n{out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    ref = sub.add_parser("references")
    ref.add_argument("--fast-dev-run", action="store_true")
    for name in ("merge", "determinism"):
        p = sub.add_parser(name)
        p.add_argument("experiment", choices=sorted(runner.EXPERIMENTS["experiments"]))
        p.add_argument("method", choices=sorted(runner.CANONICAL["methods"]))
        if name == "merge":
            p.add_argument("--fast-dev-run", action="store_true")
    args = parser.parse_args()

    if args.command == "references":
        runner.run_references(PATHS, args.fast_dev_run)
    elif args.command == "merge":
        print(runner.run_merge(args.experiment, args.method, PATHS, args.fast_dev_run))
    else:
        determinism(args.experiment, args.method)


if __name__ == "__main__":
    main()
