# Instructions: install, test and reproduce

This guide takes you from the extracted exam submission to (a) a passing test suite, (b) every
table, figure and the notebook regenerated from the shipped results, and optionally (c) a full
re-run of all experiments
on a GPU. Each step says what to run, how long it takes and what you should see.

| Level | What it shows | Needs | Time |
|---|---|---|---|
| [1. Install and test](#3-create-the-environment) | Environment, maths, DOGE behaviour, analysis are correct | CPU, internet | ~10 min |
| [2. Regenerate from shipped results](#5-regenerate-every-table-figure-and-the-notebook) | Every reported number follows from `results/raw` | CPU | ~1 min |
| [3. Re-run the experiments](#6-re-run-the-experiments-gpu) | The raw results themselves are reproducible | CUDA GPU, ~8 GB disk | 30–45 min |

Contents

1. [Requirements](#1-requirements)
2. [Get the code](#2-get-the-code)
3. [Create the environment](#3-create-the-environment)
4. [Run the test suite](#4-run-the-test-suite)
5. [Regenerate every table, figure and the notebook](#5-regenerate-every-table-figure-and-the-notebook)
6. [Re-run the experiments (GPU)](#6-re-run-the-experiments-gpu)
7. [Configuration](#7-configuration)
8. [Verify provenance](#8-verify-provenance)
9. [Result file format](#9-result-file-format)
10. [Troubleshooting](#10-troubleshooting)
11. [Clean up](#11-clean-up)

---

## 1. Requirements

| | Requirement | Notes |
|---|---|---|
| Operating system | **Linux** (x86-64) or **macOS on Apple Silicon** | Windows: use **WSL2** with Ubuntu. Intel Macs are not supported (no PyTorch 2.5 wheels). |
| Tools | `git`, `bash`, [`uv`](https://docs.astral.sh/uv/) | `uv` installs Python 3.10 for you; no system Python 3.10 needed. |
| Internet | Required for setup | Downloads Python packages and the official DOGE code; level 3 also downloads checkpoints and datasets from the Hugging Face Hub. |
| Disk | ~2 GB (macOS) or ~6 GB (Linux) for the environment | Linux PyTorch wheels include CUDA libraries. Level 3 adds ~7 GB of checkpoints and datasets. |
| Memory | 8 GB RAM for levels 1–2 | Level 3: 16 GB RAM (the four-task TIES merge peaks at 12.4 GB). |
| GPU | Only for level 3 | Canonical results used one NVIDIA RTX 4090 (24 GB). DOGE needs < 2 GB of GPU memory; evaluation a few GB. |

Install `uv` if you do not have it (see the [uv documentation](https://docs.astral.sh/uv/getting-started/installation/)), for example:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh    # or: brew install uv   /   pip install uv
```

Check:

```bash
git --version && uv --version
```

---

## 2. Get the code

Extract the submitted archive and open a terminal in the directory containing its `exam-project`
folder. No project repository URL or Git history is needed. Git is required by the setup script
to download the official DOGE dependency.

```bash
cd exam-project
```

All commands in this guide run from the repository root (`exam-project/`), which contains
`README.md`, `pyproject.toml`, and `scripts/`. If the extracted folder has another name, enter
that folder instead.

---

## 3. Create the environment

```bash
bash scripts/setup_env.sh
```

What it does, in order:

1. Clones the official DOGE implementation (a FusionBench 0.2.11 fork) into `third_party/DOGE` and
   checks out the exact commit used, `ca0a0a10c45bc0965f52f0815d6a6581ee103f85`.
2. Creates a Python **3.10** virtual environment in `.venv/` with `uv` (downloading Python 3.10 if
   needed).
3. Installs pinned dependencies: `requirements.lock` on macOS, `requirements.txt` on Linux
   (PyTorch 2.5.1, transformers 4.48.3, datasets 3.2.0, …), then the DOGE code and this project in
   editable mode.
4. Runs the 11 environment tests if the local `tests/` folder is present; otherwise it reports
   that they are skipped. It always writes an environment record to
   `results/manifests/env__<device>__<timestamp>.json`.

Expected end of the output when the local tests are present (takes 3–10 minutes, mostly downloads):

```
...........                                                              [100%]
11 passed in ...
/path/to/exam-project/results/manifests/env__cpu__<timestamp>.json
```

In a GitHub checkout, the test summary is replaced by
`Local test suite is not included; skipping environment tests.`

(`env__mps__…` on a Mac, `env__cuda__…` with an NVIDIA GPU.) The new file is your machine's record;
it is not part of the shipped evidence and can be deleted.

The script is safe to re-run: it keeps an existing environment and installs whatever is missing. To
build into another folder, pass it as an argument: `bash scripts/setup_env.sh /tmp/doge-venv`.

Activate the environment for the rest of this guide:

```bash
source .venv/bin/activate
```

---

## 4. Run the test suite

This section applies only when the local `tests/` directory is available. It is deliberately
excluded from GitHub. For a GitHub checkout, continue with
[section 5](#5-regenerate-every-table-figure-and-the-notebook); result and provenance verification
scripts are included in the published repository.

```bash
python -m pytest -q
```

Expected:

```
...................................                                      [100%]
35 passed in ...
```

The tests need no GPU and no downloads (they use the shipped results and small synthetic models).
Each test is named by a test ID from the protocol ([docs/EXPERIMENT_PROTOCOL.md](docs/EXPERIMENT_PROTOCOL.md), section 6):

| File | Tests | What it checks | Protocol IDs |
|---|---|---|---|
| `tests/test_environment.py` | 11 | Imports (including the official DOGE, TA, TIES classes), Python 3.10 / PyTorch 2.5, device selection, environment record, configurable paths | ENV-01 – ENV-05 |
| `tests/test_inventory.py` | 4 | SHA-256 of files, architecture comparison, task-vector round trip θ₀ + τ = θ, empty task vectors are flagged | CKPT-03 – CKPT-05 |
| `tests/test_baselines.py` | 8 | Task-vector subtraction, Task Arithmetic identity and formula, TIES trimming keeps exactly 20% / 30%, sign election, metrics | MATH-01, MATH-02, BASE-04 |
| `tests/test_doge.py` | 5 | Runs the **official** DOGE TA on a tiny model: no data accessed, finite objective over 400 iterations, gradient projection removes the shared-subspace component, layer-wise λ, only encoder weights change | DOGE-02, DOGE-03, DOGE-05, MATH-03 – MATH-07 |
| `tests/test_similarity.py` | 3 | The `encw-cos` key filter selects exactly the 72 CLIP ViT-B/32 matrices (84,934,656 values); cosine maths | — |
| `tests/test_analysis.py` | 4 | Two-vs-four and similarity-vs-retention maths, hypothesis rules, and that **all tables and figures regenerate from a copy of `results/raw`** and match the raw records | INT-05 |

Useful variants: `python -m pytest -v` (one line per test), `python -m pytest tests/test_doge.py`
(one file).

---

## 5. Regenerate every table, figure and the notebook

Everything here reads only `results/raw/` and runs on a CPU.

**Validate the 21 merge records** (required result fields, finite weights, no data access while
merging, save/reload check, test data loaded after merging, DOGE optimisation checks; INT-04):

```bash
python scripts/check_results.py
```

Expected: 21 lines ending in `OK`, then `0 problem(s)`.

**Regenerate all tables and figures:**

```bash
python scripts/analyze.py
```

It writes a new timestamped folder in `results/processed/analysis__<timestamp>/` (CSV and Markdown
tables, `hypotheses.md`, `summary.json`, `sources.json` with the SHA-256 of every raw file used) and
`figures/analysis__<timestamp>/` (four PNG figures), and prints the six hypothesis verdicts:

```
H1: supported
H2: supported
H3: partly supported
H4: partly supported
H5: not supported: no consistent rank relationship (descriptive)
H6: supported
```

Confirm the regenerated tables equal the shipped ones (replace `<timestamp>` with the new folder's):

```bash
for f in main_table reference_accuracy two_vs_four similarity similarity_vs_retention runtime_memory; do
  cmp results/processed/analysis__<timestamp>/$f.csv results/processed/analysis__20261018T222449Z/$f.csv && echo "$f identical"
done
```

**Rebuild and execute the notebook** (top to bottom in a fresh kernel; INT-06):

```bash
uv pip install -r requirements-notebook.txt
python scripts/build_notebook.py
```

This rewrites `notebooks/doge_merging_exam.ipynb` with fresh outputs. To explore it interactively
instead, open it in Jupyter with this environment's kernel
(`python -m ipykernel install --user --name doge-exam`, then pick **doge-exam**) and choose
**Run All**.

**A quick comparison table** of the newest record of every run:

```bash
python scripts/make_table.py            # or: python scripts/make_table.py P01 F01
```

---

## 6. Re-run the experiments (GPU)

This repeats the measurements themselves. It needs an NVIDIA GPU with CUDA (only CUDA runs are
canonical), ~8 GB of free disk for the Hugging Face cache and 30–45 minutes. On a cloud machine
(any provider), follow sections 2–4 there first; on Linux `setup_env.sh` installs the CUDA 12.4
build of PyTorch automatically.

**Write the re-run to its own results folder**, so it never mixes with the shipped results:

```bash
export DOGE_RESULTS_ROOT=$PWD/rerun/results
export DOGE_FIGURES_ROOT=$PWD/rerun/figures
nvidia-smi                       # confirm a GPU is visible
```

Then run, in this order:

| Step | Command | Time (RTX 4090) | Output |
|---|---|---|---|
| 1. Checkpoints and datasets | `python scripts/verify_inventory.py` | ~1 min + downloads | `rerun/results/manifests/inventory__*.json`; prints `PASS` for CKPT-01 – CKPT-05 and the datasets |
| 2. Reference evaluations | `python scripts/run_experiment.py references` | ~2.5 min | 5 records `ref__*` (pretrained and four experts on all four test sets) |
| 3. All 21 merges | see loop below | ~20 min | 21 records `pair__*`, `four__*` |
| 4. Similarity | `python scripts/compute_similarity.py` | < 1 min | `similarity__all-pairs__encw-cos__*.json` |
| 5. Determinism (optional) | `python scripts/run_experiment.py determinism P01 task-arithmetic` | < 1 min | prints `BASE-03 identical=True` |
| 6. Checks and analysis | `python scripts/check_results.py && python scripts/analyze.py` | < 1 min | tables and figures under `rerun/` |
| 7. Compare | `python scripts/compare_results.py rerun/results` | seconds | per-run accuracy difference and weight identity |

Run the references (step 2) before the merges: normalized retention uses the newest reference
records in the same results folder.

Step 3, all merges:

```bash
for e in P01 P02 P03 P04 P05 P06 F01; do
  for m in task-arithmetic ties doge-ta; do
    python scripts/run_experiment.py merge "$e" "$m"
  done
done
```

Experiments are defined in [`configs/experiments.yaml`](configs/experiments.yaml): P01 MNIST-SVHN,
P02 MNIST-GTSRB, P03 MNIST-EuroSAT, P04 SVHN-GTSRB, P05 SVHN-EuroSAT, P06 GTSRB-EuroSAT, F01 all four.
Methods are defined in [`configs/canonical_methods.yaml`](configs/canonical_methods.yaml). A single run,
for example `python scripts/run_experiment.py merge P01 doge-ta`, takes about one minute.

Add `--fast-dev-run` to `references` or `merge` for a one-batch smoke test (written to
`<results>/smoke/`, never mixed with complete runs).

**What to expect in step 7.** Task Arithmetic and TIES merge deterministically on the CPU, so their
merged weights should be `identical` and their accuracies equal. DOGE optimises on the GPU, so its
weights may differ in the last bits and its accuracy by a few hundredths of a point. The script exits
with status 1 if any run differs by more than `--tolerance` points (default 0.5).

**Without an NVIDIA GPU.** The same commands run on Apple Silicon (MPS) or CPU, but the records are
marked `"canonical": false` and evaluation is slow (about 9 minutes for one pair on an Apple M-series
GPU, much longer on CPU). The shipped
[reproduction record](results/manifests/reproduction__pair__mnist-svhn__task-arithmetic__mps__20261018T224347Z.json)
is such a run: weights identical to the GPU run, accuracy within 0.004 points.

---

## 7. Configuration

Storage locations default to folders inside the repository ([`configs/paths.yaml`](configs/paths.yaml))
and can be overridden per machine with a git-ignored `configs/paths.local.yaml` (same keys) or with
environment variables, which take precedence:

| Variable | Default | Purpose |
|---|---|---|
| `DOGE_RESULTS_ROOT` | `results/` | `raw/`, `processed/`, `manifests/`, `smoke/` |
| `DOGE_FIGURES_ROOT` | `figures/` | Generated figures |
| `DOGE_CACHE_ROOT` | `cache/` | Caches; the Hugging Face cache goes to `<cache_root>/huggingface` unless `HF_HOME` is set |
| `DOGE_DATA_ROOT` | `data/` | Reserved for local datasets |
| `DOGE_CHECKPOINT_ROOT` | `checkpoints/` | Reserved for local checkpoints |
| `DOGE_DEVICE` | best available | Force `cuda`, `mps` or `cpu`; an unavailable device is an error, never a silent fallback |
| `HF_TOKEN` | unset | Optional Hugging Face token if downloads are rate-limited |

The method settings, experiment matrix and pinned revisions are fixed. Changes to their values
in `configs/canonical_methods.yaml`, `configs/experiments.yaml` or `configs/pins.yaml` break the
provenance check below. The comment-only revisions recorded in
[`source_text_revisions.json`](results/manifests/source_text_revisions.json) preserve those values
and the original hashes in the submitted records.

---

## 8. Verify provenance

Each raw record stores the SHA-256 of the configuration files and runner code that produced it.
Check them against your checkout:

```bash
python scripts/verify_provenance.py
```

Expected summary:

```
 22 records: configuration matches, runner code matches
  6 records: configuration matches, runner code is an earlier version
 28 records include documented text revisions; original recorded hashes are unchanged
```

The 22 are the 21 final merge records and the similarity record. The 6 are the five reference
evaluations and the first Task Arithmetic MNIST-SVHN record, made before the data-access probe and the
save/reload check were added to the runner; the evaluation code itself did not change, and the later
Task Arithmetic re-run with the final code gave bit-identical weights and the same accuracy (93.06).
Analysis always uses the newest record of each run.

The verifier distinguishes exact byte matches from the documented text revisions listed in
[`source_text_revisions.json`](results/manifests/source_text_revisions.json). It accepts only
the specific original/current hash pairs in that manifest, not arbitrary comment or code edits.
The revisions changed comments, docstrings and one error-message label; configuration values
and executable logic are unchanged.

Also recorded per run: checkpoint revisions (`checkpoints`), library versions (`software`), GPU and
CUDA (`device`), and the DOGE code commit (`software.doge_repo_commit`).

---

## 9. Result file format

One JSON file per run in `results/raw/`, named `<run_id>__<UTC timestamp>.json`, never overwritten.
Run IDs: `pair__<task>-<task>__<method>__seed-42`, `four__mnist-svhn-gtsrb-eurosat__<method>__seed-42`,
`ref__<pretrained|expert-task>__mnist-svhn-gtsrb-eurosat`, `similarity__all-pairs__encw-cos`.

Main fields of a merge record (full contract: [protocol §7.2](docs/EXPERIMENT_PROTOCOL.md#72-minimum-result-fields)):

| Field | Meaning |
|---|---|
| `run_id`, `experiment`, `method`, `task_set`, `seed`, `status` | Identity of the run; `status` is `complete` or `failed` (failures are kept) |
| `method_config`, `config_id`, `evaluation_config` | Exact settings used |
| `checkpoints` | Hub id and revision of the pretrained model and each expert |
| `per_task`, `per_task_accuracy` | Accuracy (%), loss and number of test examples per task |
| `mean_accuracy`, `worst_task`, `worst_task_accuracy` | Summary accuracy |
| `individual_expert_accuracy`, `normalized_retention`, `mean_normalized_retention`, `interference_gap` | Relative to the individual experts (`reference_records` names the files used) |
| `merge_time_seconds`, `evaluation_time_seconds`, `peak_memory_mb`, `peak_cpu_rss_mb` | Cost (GPU peak during the merge only; Task Arithmetic and TIES merge on CPU) |
| `merged_weights_sha256`, `merged_weights_finite`, `reload_check` | Identity and integrity of the merged model |
| `data_access_during_merge` | Counts of dataset accesses while merging (all zero = data-free) |
| `merge_finished_at`, `test_data_first_loaded_at` | Proves evaluation data was loaded only after merging |
| `doge_probe` (DOGE only) | Objective per layer, projection residual, Δ share in the shared subspace |
| `device`, `software`, `provenance` | Hardware, library versions, SHA-256 of configuration and code |

---

## 10. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `uv: command not found` | Install uv (section 1) and open a new terminal. |
| `ERROR: Package 'doge-exam' requires a different Python: 3.12 not in '<3.11,>=3.10'` | `pip install` was run with the system Python. Use `bash scripts/setup_env.sh`, which creates a Python 3.10 environment with uv. |
| `fatal: detected dubious ownership in repository at '.../third_party/DOGE'` | The files belong to another user (common after copying into a container). Run `chown -R $(id -u):$(id -g) .` in the repository, or `git config --global --add safe.directory "$PWD/third_party/DOGE"`. |
| `No module named 'fusion_bench'` | The DOGE code is not installed in the active environment. Re-run `bash scripts/setup_env.sh` and `source .venv/bin/activate`. |
| Tests import the wrong environment | Run `which python`: it must point into `.venv/`. Or call `.venv/bin/python -m pytest -q` directly. |
| PyTorch install fails on an Intel Mac | PyTorch 2.5 has no Intel macOS wheels. Use Linux, WSL2 or an Apple Silicon Mac. |
| `RuntimeError: <task>: test set differs from pinned inventory` | The downloaded test set does not match the size or fingerprint verified for the shipped results (for example a different `datasets` version or a changed Hub dataset). Make sure `datasets==3.2.0` is installed, delete `cache/huggingface/datasets`, and retry. Do not edit `configs/pins.yaml` to silence it. |
| `RuntimeError: Requested device 'cuda' is not available` | `DOGE_DEVICE=cuda` is set but no GPU is visible. Check `nvidia-smi`, or unset `DOGE_DEVICE` for a non-canonical CPU/MPS run. |
| Hugging Face `429 Too Many Requests` | Set `HF_TOKEN` to a (free) Hugging Face access token and retry. |
| Matplotlib backend errors in hosted notebooks (`module://matplotlib_inline…`) | Run scripts with `MPLBACKEND=Agg`. |
| Disk full during re-runs | The cache needs ~7 GB; point `DOGE_CACHE_ROOT` (or `HF_HOME`) to a larger disk. |

---

## 11. Clean up

The environment, upstream checkout, download cache and isolated re-run output are git-ignored.
When they are no longer needed, remove them from the project root:

```bash
rm -rf .venv third_party cache rerun
```

New manifests and analysis folders created by sections 3 and 5 are **not** git-ignored. Remove
only files and timestamped folders created by your own verification run. Preserve the shipped
raw results, the four shipped experiment manifests, `source_text_revisions.json`, and the tables and figures in
`results/processed/analysis__20261018T222449Z/` and `figures/analysis__20261018T222449Z/`.
Rebuilding the notebook replaces its outputs; the submission already includes an executed copy.
