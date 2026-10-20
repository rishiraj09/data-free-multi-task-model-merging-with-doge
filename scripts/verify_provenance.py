"""Check that the configuration and code in this checkout are the ones that produced each record.

Every raw record stores the SHA-256 of the configuration files and runner code it was produced
with (`provenance`). This script compares those hashes with the working tree. Exact hash pairs
in results/manifests/source_text_revisions.json identify documented changes to comments,
docstrings and diagnostic wording; they are reported separately from exact byte matches.

Usage: python scripts/verify_provenance.py [--results results]
Exit status 1 if any record's configuration hashes do not match, or a file with a documented
text revision has changed beyond either recorded version.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def file_match(root: Path, name: str, expected: str, revisions: dict) -> str:
    actual = sha256(root / name)
    if actual == expected:
        return "exact"
    revision = revisions.get(name, {})
    if (expected == revision.get("original_sha256")
            and actual is not None and actual == revision.get("current_sha256")):
        return "documented text revision"
    return "different"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=PROJECT_ROOT / "results")
    args = parser.parse_args()

    revision_path = PROJECT_ROOT / "results/manifests/source_text_revisions.json"
    revisions = {}
    if revision_path.is_file():
        manifest = json.loads(revision_path.read_text())
        if manifest.get("schema_version") != 1:
            raise ValueError("Unsupported source text revision manifest version")
        revisions = {r["path"]: r for r in manifest["revisions"]}
    unexpected_changes = [name for name, revision in revisions.items()
                          if sha256(PROJECT_ROOT / name) not in
                          (revision["original_sha256"], revision["current_sha256"])]

    summary, config_mismatch = Counter(), []
    text_revisions = 0
    for path in sorted((args.results / "raw").glob("*.json")):
        prov = json.loads(path.read_text()).get("provenance", {})
        config_states = [file_match(PROJECT_ROOT, f, h, revisions)
                         for f, h in prov.get("config_files", {}).items()]
        code_states = [file_match(PROJECT_ROOT, f, h, revisions)
                       for f, h in prov.get("code_files", {}).items()]
        config_ok = all(state != "different" for state in config_states)
        code_ok = all(state != "different" for state in code_states)
        revised = "documented text revision" in config_states + code_states
        text_revisions += revised
        summary[(config_ok, code_ok)] += 1
        if not config_ok:
            config_mismatch.append(path.name)
        print(f"{path.name[:72]:72}  config {'match' if config_ok else 'DIFFERS'}  "
              f"code {'match' if code_ok else 'differs / earlier version'}"
              + ("  [documented text revisions]" if revised else ""))
    print()
    for (config_ok, code_ok), n in sorted(summary.items(), reverse=True):
        print(f"{n:3} records: configuration {'matches' if config_ok else 'DIFFERS'}, "
              f"runner code {'matches' if code_ok else 'is an earlier version'}")
    if text_revisions:
        print(f"{text_revisions:3} records include documented text revisions; "
              "original recorded hashes are unchanged")
    for name in unexpected_changes:
        print(f"Unrecorded change to a file with a documented text revision: {name}")
    return 1 if config_mismatch or unexpected_changes else 0


if __name__ == "__main__":
    sys.exit(main())
