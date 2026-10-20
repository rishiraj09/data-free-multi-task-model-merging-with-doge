"""Analysis computed only from raw result records (experimental protocol sections 5.3-5.5).

Pure functions over already-loaded records so they can be unit-tested; scripts/analyze.py
does the file I/O, tables and figures.
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

from scipy.stats import spearmanr

METHODS = ("task-arithmetic", "ties", "doge-ta")
METHOD_LABELS = {"task-arithmetic": "Task Arithmetic", "ties": "TIES", "doge-ta": "DOGE TA"}
TASKS = ("mnist", "svhn", "gtsrb", "eurosat")
TASK_LABELS = {"mnist": "MNIST", "svhn": "SVHN", "gtsrb": "GTSRB", "eurosat": "EuroSAT"}
FOUR = "F01"
DISSIMILAR = ("P03", "P05", "P06")  # Experimental protocol section 5.2: dissimilar task pairs.


def load_latest(raw_dir: Path) -> dict[str, tuple[Path, dict]]:
    """Newest complete, non-smoke record per run_id (file names end in a sortable timestamp)."""
    latest = {}
    for path in sorted(Path(raw_dir).glob("*.json")):
        record = json.loads(path.read_text())
        if record.get("status") == "complete" and not record.get("fast_dev_run"):
            latest[record["run_id"]] = (path, record)
    return latest


def merges_by_experiment(latest: dict) -> dict[str, dict[str, dict]]:
    out: dict[str, dict[str, dict]] = {}
    for _, record in latest.values():
        if record["kind"] == "merge":
            out.setdefault(record["experiment"], {})[record["method"]] = record
    return out


def similarity_record(latest: dict) -> dict:
    found = [r for _, r in latest.values() if r["kind"] == "similarity"]
    if len(found) != 1:
        raise ValueError(f"expected one similarity record, found {len(found)}")
    return found[0]


def pair_key(record: dict) -> str:
    return "-".join(record["task_set"])


def two_vs_four(merges: dict) -> dict:
    """Section 5.3: per task and method, F01 retention vs mean retention over the evaluated
    pairwise merges containing that task; plus the overall means."""
    pairs = {e: m for e, m in merges.items() if e != FOUR}
    result = {"per_task": {}, "overall": {}}
    for method in METHODS:
        four = merges[FOUR][method]["normalized_retention"]
        per_task = {}
        for task in TASKS:
            values = [m[method]["normalized_retention"][task] for m in pairs.values() if task in m[method]["task_set"]]
            per_task[task] = {
                "pairs_evaluated": len(values),
                "pair_mean_retention": mean(values) if values else None,
                "four_task_retention": four[task],
                "change": four[task] - mean(values) if values else None,
            }
        result["per_task"][method] = per_task
        pair_mean = mean(m[method]["mean_normalized_retention"] for m in pairs.values())
        four_mean = merges[FOUR][method]["mean_normalized_retention"]
        result["overall"][method] = {
            "pairs_evaluated": len(pairs),
            "pair_mean_retention": pair_mean,
            "four_task_mean_retention": four_mean,
            "change": four_mean - pair_mean,
        }
    return result


def similarity_vs_retention(merges: dict, cosine: dict[str, float]) -> dict:
    """Section 5.3: per method, rank the evaluated pairs by similarity and by mean retention;
    Spearman's rho is descriptive only (n = number of pairs)."""
    pairs = {e: m for e, m in merges.items() if e != FOUR}
    by_similarity = sorted(pairs, key=lambda e: -cosine[pair_key(pairs[e][METHODS[0]])])
    out = {"n_pairs": len(pairs), "order_by_similarity": by_similarity, "methods": {}}
    for method in METHODS:
        sims = [cosine[pair_key(pairs[e][method])] for e in by_similarity]
        rets = [pairs[e][method]["mean_normalized_retention"] for e in by_similarity]
        rho = spearmanr(sims, rets).statistic if len(pairs) >= 3 else None
        out["methods"][method] = {
            "order_by_retention": sorted(pairs, key=lambda e: -pairs[e][method]["mean_normalized_retention"]),
            "spearman_rho": None if rho is None else float(rho),
            "points": [
                {"experiment": e, "pair": pair_key(pairs[e][method]), "cosine": s, "mean_normalized_retention": r}
                for e, s, r in zip(by_similarity, sims, rets)
            ],
        }
    return out


def hypotheses(merges: dict, tvf: dict, svr: dict) -> dict:
    """H1-H6 (section 5.5), each with the numbers it rests on."""
    exps = sorted(merges)

    def doge_beats(metric):
        rows = {
            e: {m: merges[e][m][metric] for m in METHODS}
            for e in exps
        }
        wins = [e for e, r in rows.items() if r["doge-ta"] > r["task-arithmetic"] and r["doge-ta"] > r["ties"]]
        return rows, wins

    h1_rows, h1_wins = doge_beats("mean_accuracy")
    h2_rows, h2_wins = doge_beats("worst_task_accuracy")
    h3 = {m: tvf["overall"][m]["change"] < 0 for m in METHODS}
    h3_tasks = {m: sum(v["change"] < 0 for v in tvf["per_task"][m].values() if v["change"] is not None) for m in METHODS}
    drops = {m: -tvf["overall"][m]["change"] for m in METHODS}
    gaps = {
        e: {m: 100 - merges[e][m]["mean_normalized_retention"] for m in METHODS}
        for e in DISSIMILAR if e in merges
    }
    h6_wins = [e for e, g in gaps.items() if g["doge-ta"] < g["task-arithmetic"] and g["doge-ta"] < g["ties"]]
    rhos = {m: svr["methods"][m]["spearman_rho"] for m in METHODS}

    def verdict(ok, partial=False):
        return "supported" if ok else ("partly supported" if partial else "not supported")

    return {
        "H1": {"statement": "DOGE has higher mean accuracy than TA and TIES in each experiment",
               "experiments": len(exps), "doge_wins": h1_wins, "values": h1_rows,
               "verdict": verdict(len(h1_wins) == len(exps), bool(h1_wins))},
        "H2": {"statement": "DOGE has higher worst-task accuracy than the baselines",
               "experiments": len(exps), "doge_wins": h2_wins, "values": h2_rows,
               "verdict": verdict(len(h2_wins) == len(exps), bool(h2_wins))},
        "H3": {"statement": "Every method retains less when merging four models than two",
               "overall_lower_in_four": h3, "tasks_lower_in_four_of_4": h3_tasks,
               "verdict": verdict(all(h3.values()), any(h3.values()))},
        "H4": {"statement": "DOGE's retention drop from two to four models is smaller than the baselines'",
               "drop_points": drops,
               "smaller_than": [m for m in METHODS[:2] if drops["doge-ta"] < drops[m]],
               "verdict": verdict(all(drops["doge-ta"] < drops[m] for m in METHODS[:2]),
                                  any(drops["doge-ta"] < drops[m] for m in METHODS[:2]))},
        "H5": {"statement": "Pairs with higher task-vector cosine similarity retain more (descriptive)",
               "n_pairs": svr["n_pairs"], "spearman_rho": rhos,
               "rule": "supported if rho >= 0.5 for every method; no relationship if |rho| < 0.5 for every method",
               "verdict": "supported (descriptive)" if all(r is not None and r >= 0.5 for r in rhos.values())
               else "not supported: no consistent rank relationship (descriptive)"
               if all(r is not None and abs(r) < 0.5 for r in rhos.values())
               else "mixed (descriptive)"},
        "H6": {"statement": "DOGE loses less performance than the baselines on dissimilar pairs",
               "pairs": list(gaps), "retention_gap_points": gaps, "doge_wins": h6_wins,
               "verdict": verdict(len(h6_wins) == len(gaps) and bool(gaps), bool(h6_wins))},
    }
