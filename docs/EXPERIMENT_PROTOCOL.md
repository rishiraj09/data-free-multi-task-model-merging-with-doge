# Experimental protocol

This document describes the experimental design, evaluation criteria and verification checks.
Section numbering matches the references in the source code and result metadata. The published
checks and stored evidence described below are available without the local `tests/` directory in
[`results/`](../results/) and in [ANALYSIS_NOTES.md](ANALYSIS_NOTES.md). Unlinked repository
paths are relative to the project root. Section 9 describes the compute used
during the project (a local Mac for development, a cloud RTX 4090 for all canonical runs).

## 2. Primary Research Article

### Required starting paper

Yongxian Wei, Anke Tang, Li Shen, Zixuan Hu, Chun Yuan, and Xiaochun Cao. **Modeling Multi-Task Model Merging as Adaptive Projective Gradient Descent.** Proceedings of the 42nd International Conference on Machine Learning, PMLR 267, 2025.

- [Paper and downloadable PDF](https://proceedings.mlr.press/v267/wei25k.html)
- [Official implementation](https://github.com/WalkerWorldPeace/DOGE)

### Supporting methods

- Task Arithmetic: Ilharco et al., Editing Models with Task Arithmetic.
- TIES-Merging: Yadav et al., TIES-Merging: Resolving Interference When Merging Models.

### Relevant paper settings

The paper's primary vision benchmark uses CLIP ViT-B/32 and eight tasks. It reports DOGE applied to Task Arithmetic as DOGETA. The paper uses:

- Python 3.10
- PyTorch 2.5
- 400 optimization iterations for the modification vector
- Learning rate 1e-4
- Vision global scaling factor eta = 0.07
- Retention of the top 30 percent of parameters by magnitude inside DOGE ("following TIES"; official `K: 30`). TIES itself uses the official `threshold: 20` (see [paper-to-code map, D3](PAPER_TO_CODE_MAP.md#5-paper-versus-code-discrepancies))
- Top-1 accuracy for vision evaluation

The official implementation and configuration files define the experimental settings.
The values used in the submitted runs and their rationale are documented in the
[paper-to-code map](PAPER_TO_CODE_MAP.md#6-canonical-settings-canonical-v1).

---

## 5. Exam Experimental Design

### 5.1 Task set

The evaluated tasks are:

1. MNIST
2. SVHN
3. GTSRB
4. EuroSAT

Reasons:

- They are included in the paper's CLIP vision benchmark (the eight Task Arithmetic tasks).
- They cover handwritten digits, street-view digits, traffic signs, and satellite imagery.
- They span related pairs (MNIST-SVHN) and clearly dissimilar pairs (MNIST-EuroSAT).
- Their datasets are more manageable than the largest benchmark tasks.

Checkpoint and dataset availability was verified in the
[submitted inventory](../results/manifests/inventory__20261018T213330Z.json); the task set is final.

### 5.2 Experiment matrix

| ID | Tasks | Expected relationship |
|---|---|---|
| P01 | MNIST, SVHN | Relatively related digit recognition |
| P02 | MNIST, GTSRB | Different domains, both symbol-like |
| P03 | MNIST, EuroSAT | Dissimilar |
| P04 | SVHN, GTSRB | Street imagery, symbol recognition |
| P05 | SVHN, EuroSAT | Dissimilar |
| P06 | GTSRB, EuroSAT | Dissimilar |
| F01 | MNIST, SVHN, GTSRB, EuroSAT | Four-task merge |

Each task set was merged with Task Arithmetic, TIES-Merging and DOGE applied to Task Arithmetic.
Task-vector cosine similarity was computed for all six pairs. Reference evaluations test the
pretrained model and all four individual experts on all four test sets.

The submitted experiments use one frozen configuration per method; hyperparameter sweeps and
ablations are outside the project scope. All 21 merge runs and all six pairwise similarities
were completed.

### 5.3 Analysis definitions

- Two-versus-four comparison: for each task, compare its normalized retention in F01 with its mean normalized retention across the evaluated pairwise merges that contain it, per method. Also compare mean normalized retention of F01 with the mean over evaluated pairs. All six pairwise merges were completed, so every task has three pairwise results for each method.
- Similarity analysis: per method, rank the evaluated pairs by task-vector cosine similarity and by mean normalized retention, and report both orderings plus Spearman's rank coefficient as a descriptive statistic only. The exact similarity definition, including the parameter set used to flatten the vectors, was fixed before inspecting merge results; see the [task-vector definition](PAPER_TO_CODE_MAP.md#7-task-vector-similarity-definition-encw-cos).

### 5.4 Metrics

For every run, record:

- Per-task top-1 accuracy
- Mean accuracy
- Worst-task accuracy
- Individual-expert accuracy
- Performance-retention ratio per task:
  100 times merged accuracy divided by individual-expert accuracy
- Mean normalized performance retention
- Interference gap:
  individual-expert accuracy minus merged accuracy
- Improvement of DOGE over Task Arithmetic
- Improvement of DOGE over TIES
- Task-vector cosine similarity for every pair
- Change in normalized retention from two-model to four-model merging
- Wall-clock merging time
- Evaluation time
- Peak GPU memory
- Hardware, device, precision, seed, commit hash, and configuration identifier

### 5.5 Primary hypotheses

- H1: DOGE has higher mean accuracy than Task Arithmetic and TIES in each experiment.
- H2: DOGE has higher worst-task accuracy than the baselines.
- H3: Every method retains less individual-expert performance when merging four models than when merging two.
- H4: DOGE's retention drop from two to four models is smaller than the baselines' drop.
- H5: Pairs with higher task-vector cosine similarity retain more individual-expert performance (descriptive, across evaluated pairs).
- H6: DOGE loses less performance than the baselines on dissimilar pairs.

Failure to confirm a hypothesis is a valid result and must not be hidden.

---

## 6. Verification and published evidence

The research design and result checks remain relevant without publishing the local unit tests.
The GitHub repository includes the commands and evidence below; none requires `tests/` or
`pytest`. The setup script skips its optional environment-test step when that directory is
absent. Additional synthetic unit tests are retained locally and are not part of the public
reproduction instructions. Identifiers such as ENV, CKPT, MATH, BASE, DOGE and INT in source
comments identify the verification categories described here.

### 6.1 Environment records

The environment record captures Python and package versions, CUDA/MPS/CPU selection, GPU details
and the official DOGE commit (ENV-01 to ENV-05). The setup script records the current environment;
it can also be recorded explicitly after activating the project environment:

```bash
python scripts/record_env.py
```

The [submitted CUDA environment record](../results/manifests/env__cuda__20261018T211911Z.json)
describes the RTX 4090 machine used for the canonical measurements. A new environment record
identifies the reproduction machine and does not replace the submitted one.

### 6.2 Checkpoint and dataset inventory

The [submitted inventory](../results/manifests/inventory__20261018T213330Z.json) contains the
pretrained/expert checkpoint revisions, file checksums, architecture compatibility, task-vector
round-trip checks, dataset sizes/fingerprints and preprocessing settings (CKPT-01 to CKPT-05).
These checks can be repeated with the pinned downloads:

```bash
python scripts/verify_inventory.py
```

This command downloads models and datasets and is separate from reanalysing the stored results.
The runner also rejects test splits whose size or fingerprint differs from
[`configs/pins.yaml`](../configs/pins.yaml).

### 6.3 Numerical checks and tolerances

Raw DOGE records contain probe diagnostics for finite objectives and corrections, projected
gradient residuals, projection idempotence, layer counts and iteration counts. The published
result checker enforces finite objective/correction values, 400 iterations on 72 layers and
relative projected-gradient residual at most 1e-3 (DOGE-02, DOGE-03 and MATH-07).
The [paper-to-code map](PAPER_TO_CODE_MAP.md#5-paper-versus-code-discrepancies) explains the
float32 tolerance and why the gradient constraint does not imply the same constraint on Adam's
accumulated correction.

The additional synthetic mathematical tests (MATH-01 to MATH-06) are local only. Their float32
absolute/relative tolerance is 1e-5 unless a documented numerical reason justifies a relaxation;
they are not needed to run the public result checks.

### 6.4 Reference evaluations and determinism

The five reference records evaluate the pretrained model and each expert on all four test sets.
The notebook compares their own-task accuracies with the paper-reported values in
[`configs/paper_reference.yaml`](../configs/paper_reference.yaml) (BASE-01 and BASE-02).
Preprocessing, class heads and test examples are shared across methods (BASE-05).
The [method configuration](../configs/canonical_methods.yaml) fixes the TIES trim fraction and
other merging settings (BASE-04).

The [Task Arithmetic repeat record](../results/manifests/determinism__pair__mnist-svhn__task-arithmetic__seed-42__20261018T215202Z.json)
contains two identical merged-weight hashes (BASE-03). The
[separate MPS reproduction](../results/manifests/reproduction__pair__mnist-svhn__task-arithmetic__mps__20261018T224347Z.json)
provides a second-machine comparison. A new repeat can be run with:

```bash
python scripts/run_experiment.py determinism P01 task-arithmetic
```

This repeats GPU/model evaluation work; use the isolated results directories described in
[INSTRUCTION.md, section 6](../INSTRUCTION.md#6-re-run-the-experiments-gpu).

### 6.5 Merge validity checks

After installation, validate all 21 newest complete merge records without rerunning any merge:

```bash
python scripts/check_results.py
```

The expected summary is `0 problem(s)`. Checks cover required result fields, finite merged
weights, zero dataset access during merging, a passing checkpoint save/reload comparison,
and test data first loaded after the merge finished. DOGE records also receive the numerical
checks above (DOGE-02 to DOGE-06 and INT-04). The notebook displays the check result for every
experiment/method combination.

A new run writes a timestamped record. The runner does not automatically skip existing runs or
resume a partially completed merge. To continue an interrupted batch, run the missing task-set
and method combinations explicitly; analysis selects the newest complete, non-smoke record
for each run identifier.

### 6.6 Analysis, notebook and provenance checks

Regenerate the analysis from the included raw records and verify their recorded sources:

```bash
python scripts/verify_provenance.py
python scripts/analyze.py
uv pip install --python .venv/bin/python -r requirements-notebook.txt
python scripts/build_notebook.py
```

These commands need no model downloads or GPU after the analysis environment is installed.
The notebook runs top to bottom and includes tables, plots and validation results (INT-05 and
INT-06). Rebuilding replaces its saved outputs. Each analysis folder includes `sources.json`,
which records the SHA-256 of the raw files used. See
[INSTRUCTION.md, section 5](../INSTRUCTION.md#5-regenerate-every-table-figure-and-the-notebook)
for comparisons against the included tables and
[section 8](../INSTRUCTION.md#8-verify-provenance) for the original hashes and documented text
revisions accepted by the provenance verifier.

### 6.7 Experimental validity

- Every method uses the same test splits, preprocessing and class heads.
- The base model and expert checkpoint revisions are fixed.
- Each method has one configuration across two-task and four-task merges; coefficients are not
  tuned on final test accuracy.
- Exact task sets, hardware, precision, settings and raw metrics are recorded.
- Seed 42 is fixed. The Task Arithmetic repeat is documented; multi-seed variability and
  cross-hardware DOGE variability are not measured.
- Similarity analysis uses six pairs and is descriptive. Task identity and repeated tasks across
  pairs limit the interpretation of rank correlations.
- Tables and figures are generated from the raw records. Unsupported hypotheses and limitations
  are reported in [ANALYSIS_NOTES.md](ANALYSIS_NOTES.md).

### 6.8 Included evidence

| Evidence | Location and coverage |
|---|---|
| Complete merge results | [`results/raw/`](../results/raw/): all six pairs and F01, each with three methods |
| Reference evaluations | Five records in [`results/raw/`](../results/raw/) |
| Pairwise similarity | [All six pairs](../results/raw/similarity__all-pairs__encw-cos__20261018T220404Z.json) |
| Run environment and inventory | [`results/manifests/`](../results/manifests/) |
| Derived tables and source hashes | [Submitted analysis folder](../results/processed/analysis__20261018T222449Z/) |
| Plots | [Submitted figures](../figures/analysis__20261018T222449Z/) |
| Executed walkthrough and record checks | [Jupyter notebook](../notebooks/doge_merging_exam.ipynb) |

The local `tests/` directory and presentation materials are not included in the GitHub repository.

---

## 7. Result Artifact Contract

### 7.1 Required directory convention

The [README repository layout](../README.md#repository-layout) describes the submitted files.
Raw results live in [`results/raw/`](../results/raw/), derived tables in
[`results/processed/`](../results/processed/), run manifests in
[`results/manifests/`](../results/manifests/), and figures in [`figures/`](../figures/).

Large datasets, caches, and checkpoints must not be committed.

### 7.2 Minimum result fields

Merge runs save JSON records with the fields below. Logical fields may be nested: the project
commit is `provenance.git_commit` (`null` in the submitted records), checkpoint identifiers are
in `checkpoints`, configuration values are in `method_config` and `evaluation_config`, and
configuration paths and hashes are in `provenance.config_files`. See the
[field mapping in INSTRUCTION.md section 9](../INSTRUCTION.md#9-result-file-format).

- run_id
- timestamp
- git_commit
- method
- task_set
- task_count
- seed
- base_model
- checkpoint identifiers
- configuration path
- device
- precision
- software versions
- per_task_accuracy
- mean_accuracy
- worst_task_accuracy
- normalized_retention
- merge_time_seconds
- evaluation_time_seconds
- peak_memory_mb
- status
- notes

### 7.3 Naming convention

Use stable, sortable identifiers such as:

    pair__mnist-svhn__doge-ta__seed-42
    pair__mnist-eurosat__ties__seed-42
    four__mnist-svhn-gtsrb-eurosat__task-arithmetic__seed-42

Task order inside an identifier follows the task list in section 5.1: MNIST, SVHN, GTSRB, EuroSAT.
Similarity results use `similarity__all-pairs__<definition-id>`. Every result filename appends
`__<UTC timestamp>.json` to its run identifier; for example, the
[submitted similarity record](../results/raw/similarity__all-pairs__encw-cos__20261018T220404Z.json).

---

## 9. Compute and Reproducibility Strategy

### Local M4 MacBook Air

Use for:

- Reading, development, unit tests, plotting, and documentation
- CPU smoke tests
- Optional MPS inference experiments where supported

Do not rely on MPS for the canonical final result because the official project is CUDA-oriented and some linear-algebra behavior may differ.

### CUDA runtime used for the submitted results

All canonical reference evaluations and merge experiments ran on one Runpod RTX 4090 (24 GB),
with Python 3.10.18, PyTorch 2.5.1 and CUDA 12.4. The exact environment is in the
[submitted environment record](../results/manifests/env__cuda__20261018T211911Z.json).
Colab was considered initially, but its runtime had no GPU; it was not used for the final results.

The pod had no persistent volume. Results were copied to the local machine after each run;
the pod has since been stopped. Its `/workspace` paths are historical machine locations, not
paths to files included in this submission.

For reproduction, use a fresh CUDA machine and follow
[INSTRUCTION.md section 6](../INSTRUCTION.md#6-re-run-the-experiments-gpu). The setup script creates the
pinned Python environment, and `scripts/run_experiment.py` builds the required pools directly.
Choose storage appropriate to the runtime and keep independent copies of results. Use separate
`DOGE_RESULTS_ROOT` and `DOGE_FIGURES_ROOT` directories for re-runs so the submitted evidence
is preserved. A 24 GB GPU was sufficient for the measured runs.

### Reproducibility requirements

- Fixed seeds where stochasticity exists
- Exact configuration saved with every run
- Configuration and runner SHA-256 hashes saved per run; `provenance.git_commit` is `null` in
  the submitted records, while the official DOGE commit is recorded in `software.doge_repo_commit`
- Software versions recorded
- Hardware and precision recorded
- Raw and processed results separated
- Figures generated programmatically from raw results
- No secrets, tokens, datasets, or large checkpoints committed
