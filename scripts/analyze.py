"""Regenerate every analysis table and figure from results/raw only (INT-05).

Writes results/processed/analysis__<timestamp>/ (CSV and Markdown tables, hypotheses,
summary.json, sources.json with the SHA-256 of every raw file used) and
figures/analysis__<timestamp>/ (PNG figures). Nothing is typed in by hand.

Usage: python scripts/analyze.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from doge_exam import analysis as A  # noqa: E402
from doge_exam.paths import load_paths  # noqa: E402

# Light-mode palette; colors 1-3 validated pairwise for CVD.
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
COLORS = {"task-arithmetic": "#2a78d6", "ties": "#eb6834", "doge-ta": "#1baf7a"}
BLUES = LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
)

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "text.color": INK, "axes.titlecolor": INK, "font.size": 10, "axes.titlesize": 11,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False, "figure.dpi": 150,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.15,
})


def write_table(rows: list[dict], path: Path, note: str) -> None:
    columns = list(dict.fromkeys(k for r in rows for k in r))
    with path.with_suffix(".csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)

    def fmt(v):
        return "" if v is None else f"{v:.2f}" if isinstance(v, float) else str(v)

    md = [note, "", "| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    md += ["| " + " | ".join(fmt(r.get(c)) for c in columns) + " |" for r in rows]
    path.with_suffix(".md").write_text("\n".join(md) + "\n")


def experiment_order(merges):
    return sorted((e for e in merges if e != A.FOUR)) + [A.FOUR]


def main_table(merges, refs):
    rows = []
    for e in experiment_order(merges):
        for m in A.METHODS:
            r = merges[e][m]
            row = {"experiment": e, "tasks": "-".join(A.TASK_LABELS[t] for t in r["task_set"]), "method": A.METHOD_LABELS[m]}
            row |= {A.TASK_LABELS[t]: r["per_task_accuracy"].get(t) for t in A.TASKS}
            row |= {
                "mean": r["mean_accuracy"],
                "worst_task": A.TASK_LABELS[r["worst_task"]],
                "worst": r["worst_task_accuracy"],
                "mean_norm_retention": r["mean_normalized_retention"],
            }
            if m == "doge-ta":
                row["doge_minus_ta"] = r["mean_accuracy"] - merges[e]["task-arithmetic"]["mean_accuracy"]
                row["doge_minus_ties"] = r["mean_accuracy"] - merges[e]["ties"]["mean_accuracy"]
            rows.append(row)
    ref_rows = [
        {"model": "pretrained" if k == "pretrained" else f"{A.TASK_LABELS[k.split('-')[1]]} expert",
         **{A.TASK_LABELS[t]: refs[k]["per_task_accuracy"][t] for t in A.TASKS}}
        for k in ["pretrained", *(f"expert-{t}" for t in A.TASKS)]
    ]
    return rows, ref_rows


def fig_per_task(merges, refs, out):
    rows = [(e, t) for e in experiment_order(merges) for t in merges[e]["task-arithmetic"]["task_set"]]
    fig, ax = plt.subplots(figsize=(7.5, 0.34 * len(rows) + 1.4))
    y = np.arange(len(rows))[::-1]
    for yi, (e, t) in zip(y, rows):
        expert = refs[f"expert-{t}"]["per_task_accuracy"][t]
        vals = [merges[e][m]["per_task_accuracy"][t] for m in A.METHODS]
        ax.plot([min(vals), expert], [yi, yi], color=GRID, lw=2, zorder=1)
        ax.plot([expert, expert], [yi - 0.32, yi + 0.32], color=INK, lw=2, zorder=2)
        for m, v in zip(A.METHODS, vals):
            ax.scatter(v, yi, s=48, color=COLORS[m], edgecolor=SURFACE, linewidth=1.5, zorder=4)
    ax.set_yticks(y, [f"{e} · {A.TASK_LABELS[t]}" for e, t in rows])
    ax.set_xlabel("Test accuracy (%)")
    ax.set_xlim(65, 101)
    ax.grid(axis="y", visible=False)
    handles = [plt.Line2D([], [], marker="o", ls="", color=COLORS[m], markersize=7, label=A.METHOD_LABELS[m]) for m in A.METHODS]
    handles.append(plt.Line2D([], [], marker="|", ls="", color=INK, markersize=10, markeredgewidth=2, label="Individual expert"))
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=4, fontsize=9)
    ax.set_title("Per-task accuracy of each merged model", loc="left", pad=28)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_two_vs_four(tvf, out):
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), sharey=True, sharex=True)
    labels = [A.TASK_LABELS[t] for t in A.TASKS] + ["Mean"]
    y = np.arange(len(labels))[::-1]
    for ax, m in zip(axes, A.METHODS):
        two = [tvf["per_task"][m][t]["pair_mean_retention"] for t in A.TASKS] + [tvf["overall"][m]["pair_mean_retention"]]
        four = [tvf["per_task"][m][t]["four_task_retention"] for t in A.TASKS] + [tvf["overall"][m]["four_task_mean_retention"]]
        for yi, a, b in zip(y, two, four):
            ax.plot([a, b], [yi, yi], color=COLORS[m], lw=2, alpha=0.45, zorder=1)
            ax.scatter(a, yi, s=56, facecolor=SURFACE, edgecolor=COLORS[m], linewidth=2, zorder=3)
            ax.scatter(b, yi, s=70, color=COLORS[m], edgecolor=SURFACE, linewidth=1.5, zorder=4)
            ax.annotate(f"{b - a:+.1f}", (min(a, b), yi), xytext=(-6, 0), textcoords="offset points",
                        ha="right", va="center", fontsize=8, color=INK_2)
        ax.axhline(0.5, color=AXIS, lw=0.8)
        ax.set_title(A.METHOD_LABELS[m], loc="left")
        ax.set_yticks(y, labels)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("Normalized retention (%)")
    axes[0].set_xlim(60, 102)
    handles = [plt.Line2D([], [], marker="o", ls="", mfc=SURFACE, mec=MUTED, mew=2, markersize=7, label="Two tasks (mean over pairs with the task)"),
               plt.Line2D([], [], marker="o", ls="", color=MUTED, markersize=7, label="Four tasks (F01)")]
    fig.legend(handles=handles, loc="upper left", ncol=2, bbox_to_anchor=(0.01, 1.02), fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out)
    plt.close(fig)


def fig_similarity_matrix(cosine, out):
    n = len(A.TASKS)
    mat = np.full((n, n), np.nan)
    for i, a in enumerate(A.TASKS):
        for j, b in enumerate(A.TASKS):
            key = f"{a}-{b}" if f"{a}-{b}" in cosine else f"{b}-{a}"
            if i != j:
                mat[i, j] = cosine[key]
    fig, ax = plt.subplots(figsize=(4.8, 4.0))
    vmax = np.nanmax(mat)
    im = ax.imshow(np.ma.masked_invalid(mat), cmap=BLUES, vmin=0, vmax=vmax)
    for i in range(n):
        for j in range(n):
            text = "—" if i == j else f"{mat[i, j]:.3f}"
            dark = i != j and mat[i, j] > 0.55 * vmax
            ax.text(j, i, text, ha="center", va="center", fontsize=9, color=SURFACE if dark else INK)
    labels = [A.TASK_LABELS[t] for t in A.TASKS]
    ax.set_xticks(range(n), labels)
    ax.set_yticks(range(n), labels)
    ax.grid(False)
    ax.set_xticks(np.arange(-0.5, n), minor=True)
    ax.set_yticks(np.arange(-0.5, n), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)  # 2px surface gap between cells
    ax.tick_params(which="minor", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.outline.set_visible(False)
    cb.set_label("Cosine similarity (encw-cos)", color=INK_2)
    ax.set_title("Task-vector cosine similarity", loc="left")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_similarity_vs_retention(svr, out):
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), sharex=True)
    for ax, m in zip(axes, A.METHODS):
        pts = svr["methods"][m]["points"]
        xs = [p["cosine"] for p in pts]
        ys = [p["mean_normalized_retention"] for p in pts]
        ax.scatter(xs, ys, s=56, color=COLORS[m], edgecolor=SURFACE, linewidth=1.5, zorder=3)
        # P04, P05 and P06 sit close together: P04 is labelled to its right, and of P05/P06
        # the higher point is labelled above, the lower one below.
        ret = {p["experiment"]: p["mean_normalized_retention"] for p in pts}
        lower = min(("P05", "P06"), key=ret.get)
        for p in pts:
            name = "-".join(A.TASK_LABELS[t][:2] if t != "eurosat" else "Eu" for t in p["pair"].split("-"))
            offset = (8, -3) if p["experiment"] == "P04" else (5, -11) if p["experiment"] == lower else (5, 4)
            ax.annotate(f"{p['experiment']} {name}", (p["cosine"], p["mean_normalized_retention"]),
                        xytext=offset, textcoords="offset points", fontsize=7.5, color=INK_2)
        rho = svr["methods"][m]["spearman_rho"]
        ax.set_title(f"{A.METHOD_LABELS[m]}   Spearman ρ = {rho:+.2f} (n = {svr['n_pairs']})", loc="left", fontsize=10)
        ax.set_xlabel("Task-vector cosine similarity")
        pad = (max(ys) - min(ys)) * 0.25 + 0.5
        ax.set_ylim(min(ys) - pad, max(ys) + pad)
    axes[0].set_ylabel("Mean normalized retention (%)")
    axes[0].set_xlim(0, max(p["cosine"] for p in svr["methods"][A.METHODS[0]]["points"]) * 1.35)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def hypotheses_md(h: dict) -> str:
    lines = ["# Hypothesis assessment (generated from results/raw by scripts/analyze.py)", ""]
    for key, v in h.items():
        lines.append(f"## {key}: {v['verdict']}")
        lines.append(f"{v['statement']}.")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps({k: x for k, x in v.items() if k not in ("statement", "verdict")}, indent=2))
        lines.append("```")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    paths = load_paths()
    latest = A.load_latest(paths.raw_results)
    merges = A.merges_by_experiment(latest)
    refs = {rid.split("__")[1]: rec for rid, (_, rec) in latest.items() if rec["kind"] == "reference"}
    cosine = A.similarity_record(latest)["cosine_similarity"]

    tvf = A.two_vs_four(merges)
    svr = A.similarity_vs_retention(merges, cosine)
    hyp = A.hypotheses(merges, tvf, svr)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    tables = paths.processed_results / f"analysis__{stamp}"
    figures = paths.figures_root / f"analysis__{stamp}"
    tables.mkdir(parents=True)
    figures.mkdir(parents=True)
    note = "Generated by scripts/analyze.py from results/raw (accuracy and retention in %)."

    main_rows, ref_rows = main_table(merges, refs)
    write_table(main_rows, tables / "main_table", note)
    write_table(ref_rows, tables / "reference_accuracy", note)
    write_table(
        [{"method": A.METHOD_LABELS[m], "task": A.TASK_LABELS[t], **v} for m in A.METHODS for t, v in tvf["per_task"][m].items()]
        + [{"method": A.METHOD_LABELS[m], "task": "mean", **v} for m, v in tvf["overall"].items()],
        tables / "two_vs_four", note + " pair_mean_* = mean over evaluated pairwise merges containing the task.",
    )
    write_table([{"pair": k, "cosine": v} for k, v in cosine.items()], tables / "similarity", note)
    write_table(
        [{"method": A.METHOD_LABELS[m], **p, "rank_similarity": svr["order_by_similarity"].index(p["experiment"]) + 1,
          "rank_retention": svr["methods"][m]["order_by_retention"].index(p["experiment"]) + 1}
         for m in A.METHODS for p in svr["methods"][m]["points"]],
        tables / "similarity_vs_retention", note + " Spearman's rho is descriptive only (n = 6).",
    )
    write_table(
        [{"experiment": e, "method": A.METHOD_LABELS[m], "task_count": merges[e][m]["task_count"],
          "merge_time_s": merges[e][m]["merge_time_seconds"], "evaluation_time_s": merges[e][m]["evaluation_time_seconds"],
          "peak_gpu_mb": merges[e][m]["peak_memory_mb"], "peak_cpu_rss_mb": merges[e][m]["peak_cpu_rss_mb"]}
         for e in experiment_order(merges) for m in A.METHODS],
        tables / "runtime_memory",
        note + " GPU peak covers method.run only; Task Arithmetic and TIES merge on CPU (0 MB GPU). "
        "CPU RSS is the process peak up to the end of merging. Hardware: one RTX 4090.",
    )
    (tables / "hypotheses.md").write_text(hypotheses_md(hyp))

    fig_per_task(merges, refs, figures / "per_task_accuracy.png")
    fig_two_vs_four(tvf, figures / "two_vs_four_retention.png")
    fig_similarity_matrix(cosine, figures / "similarity_matrix.png")
    fig_similarity_vs_retention(svr, figures / "similarity_vs_retention.png")

    root = paths.results_root.parent
    sources = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p, _ in sorted(latest.values())}
    (tables / "sources.json").write_text(json.dumps(sources, indent=2))
    summary = {"generated_at": stamp, "two_vs_four": tvf, "similarity_vs_retention": svr, "hypotheses": hyp,
               "tables": str(tables.relative_to(root)), "figures": str(figures.relative_to(root))}
    (tables / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(tables, figures, sep="\n")
    for k, v in hyp.items():
        print(f"{k}: {v['verdict']}")


if __name__ == "__main__":
    main()
