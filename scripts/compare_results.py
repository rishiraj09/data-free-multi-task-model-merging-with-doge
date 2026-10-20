"""Compare a re-run against the shipped results, run by run.

For every run_id present in both results roots (newest complete record of each), prints the
mean accuracy of both, their difference, and whether the merged weights are bit-identical.
Use it after re-running experiments into a separate results root (see INSTRUCTION.md).

Usage: python scripts/compare_results.py <rerun-results-root> [--reference results]
Exit status 1 if any shared run differs by more than --tolerance points (default 0.5).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from doge_exam.analysis import load_latest
from doge_exam.paths import PROJECT_ROOT


def summarize(root: Path) -> dict[str, dict]:
    return {rid: rec for rid, (_, rec) in load_latest(root / "raw").items() if "mean_accuracy" in rec}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("rerun", type=Path, help="results root of the re-run (contains raw/)")
    parser.add_argument("--reference", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument("--tolerance", type=float, default=0.5, help="max |difference| in accuracy points")
    args = parser.parse_args()

    ref, new = summarize(args.reference), summarize(args.rerun)
    shared = sorted(set(ref) & set(new))
    if not shared:
        print("No run present in both results roots.")
        return 1
    worst = 0.0
    print(f"{'run_id':62} {'shipped':>8} {'rerun':>8} {'diff':>7}  weights")
    for rid in shared:
        a, b = ref[rid], new[rid]
        diff = b["mean_accuracy"] - a["mean_accuracy"]
        worst = max(worst, abs(diff))
        same = a.get("merged_weights_sha256") == b.get("merged_weights_sha256")
        weights = "-" if "merged_weights_sha256" not in a else "identical" if same else "different"
        print(f"{rid:62} {a['mean_accuracy']:8.2f} {b['mean_accuracy']:8.2f} {diff:+7.3f}  {weights}")
    missing = sorted(set(ref) - set(new))
    print(f"\n{len(shared)} runs compared; largest |difference| {worst:.3f} points; "
          f"{len(missing)} shipped runs not re-run.")
    return 0 if worst <= args.tolerance else 1


if __name__ == "__main__":
    sys.exit(main())
