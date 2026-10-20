"""Task-vector cosine similarity for all six pairs of the four tasks (`encw-cos`).

Uses the pinned checkpoints (configs/pins.yaml); no data and no evaluation are involved.
Writes results/raw/similarity__all-pairs__encw-cos__<timestamp>.json.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone

from doge_exam.paths import configure_hf_cache, load_paths

PATHS = load_paths()
PATHS.ensure()
configure_hf_cache(PATHS)

from transformers import CLIPVisionModel  # noqa: E402

from doge_exam import runner, similarity  # noqa: E402


def main() -> None:
    start = time.time()
    tasks = runner.EXPERIMENTS["tasks"]
    load = lambda repo: CLIPVisionModel.from_pretrained(runner.local_model(repo)).state_dict()  # noqa: E731
    theta_0 = load(runner.CANONICAL["base_model"])
    keys = similarity.selected_keys(theta_0)
    vectors = {t: similarity.task_vector(theta_0, load(runner.CANONICAL["experts"][t])) for t in tasks}
    record = {
        "run_id": f"similarity__all-pairs__{similarity.DEFINITION_ID}",
        "kind": "similarity",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "definition_id": similarity.DEFINITION_ID,
        "definition": runner.CANONICAL["similarity"],
        "num_matrices": len(keys),
        "num_values": int(next(iter(vectors.values())).numel()),
        "checkpoints": runner.checkpoint_ids(tasks),
        "task_vector_l2_norm": {t: v.norm().item() for t, v in vectors.items()},
        "cosine_similarity": similarity.pairwise_cosine(vectors),
        "elapsed_seconds": round(time.time() - start, 1),
        "provenance": runner.provenance()
        | {"similarity_code_sha256": runner._sha256_text(runner.PROJECT_ROOT / "src/doge_exam/similarity.py")},
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = PATHS.raw_results / f"{record['run_id']}__{stamp}.json"
    out.write_text(json.dumps(record, indent=2))
    for pair, value in record["cosine_similarity"].items():
        print(f"{pair}: {value:.4f}")
    print(out)


if __name__ == "__main__":
    main()
