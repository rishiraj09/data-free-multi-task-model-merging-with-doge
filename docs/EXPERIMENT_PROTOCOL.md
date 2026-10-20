# Experimental protocol

This document describes the experimental design, evaluation criteria and verification checks.
Section numbering matches the references in the source code and result metadata. Status
checkboxes record what was verified; the evidence for each item is in
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
- LiNeS may be discussed as related work but is not required in the reduced experimental matrix.

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

The required tasks are:

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

| ID | Tasks | Expected relationship | Tier |
|---|---|---|---|
| P01 | MNIST, SVHN | Relatively related digit recognition | 1 |
| P03 | MNIST, EuroSAT | Dissimilar | 1 |
| F01 | MNIST, SVHN, GTSRB, EuroSAT | Four-task merge | 1 |
| P02 | MNIST, GTSRB | Different domains, both symbol-like | 3 |
| P04 | SVHN, GTSRB | Street imagery, symbol recognition | 3 |
| P05 | SVHN, EuroSAT | Dissimilar | 3 |
| P06 | GTSRB, EuroSAT | Dissimilar | 3 |

For every experiment, run:

- Task Arithmetic
- TIES-Merging
- DOGE applied to Task Arithmetic

Tier 2 computes task-vector cosine similarity for all six pairs, P01 to P06, regardless of whether their merges are run.

Also evaluate the pretrained model and all four individual experts on all four test sets as references.

The submitted experiments use one frozen configuration per method; hyperparameter sweeps and
ablations are outside the project scope. All 21 merge runs and all six pairwise similarities
were completed.

### 5.3 Analysis definitions

- Two-versus-four comparison: for each task, compare its normalized retention in F01 with its mean normalized retention across the evaluated pairwise merges that contain it, per method. Also compare mean normalized retention of F01 with the mean over evaluated pairs. With Tier 1 only, GTSRB has no pairwise result; report its F01 value alone and state this limitation.
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

## 6. Tests and verification

The Python test suite described below is retained locally and excluded from GitHub. The
published repository includes the executed notebook, result checks and provenance verifier;
see [INSTRUCTION.md §5](../INSTRUCTION.md#5-regenerate-every-table-figure-and-the-notebook)
and [§8](../INSTRUCTION.md#8-verify-provenance).

### 6.1 Environment tests

| Test ID | Test | Pass condition |
|---|---|---|
| ENV-01 | Import core dependencies | Imports complete without exceptions |
| ENV-02 | Detect device | CUDA device is reported on Colab/Runpod; CPU or MPS fallback is explicit locally |
| ENV-03 | Record versions | Python, PyTorch, CUDA, transformers, and FusionBench versions are written to an artifact |
| ENV-04 | Clean restart | Environment setup works in a fresh runtime |
| ENV-05 | Storage paths | Dataset, checkpoint, cache, result, and figure paths are configurable |

### 6.2 Checkpoint integrity tests

| Test ID | Test | Pass condition |
|---|---|---|
| CKPT-01 | Pretrained checkpoint loads | No missing or unexpected keys unless documented |
| CKPT-02 | Expert checkpoints load | All four required experts load successfully |
| CKPT-03 | Architecture consistency | Parameter keys and shapes match the pretrained model |
| CKPT-04 | Checksum inventory | Checkpoint names, sizes, and hashes are recorded |
| CKPT-05 | Task-vector round trip | theta_0 plus tau_i reconstructs theta_i within numerical tolerance |

### 6.3 Mathematical unit tests

| Test ID | Test | Pass condition |
|---|---|---|
| MATH-01 | Task-vector subtraction | Known synthetic tensors produce expected differences |
| MATH-02 | Task Arithmetic identity | Zero task vectors reproduce the pretrained model |
| MATH-03 | SVD basis orthogonality | B transpose B is approximately the identity |
| MATH-04 | Projection idempotence | Projecting twice matches projecting once |
| MATH-05 | Orthogonal residual | Shared basis transpose times projected residual is approximately zero |
| MATH-06 | Adaptive coefficient shape | One valid coefficient exists for every required task/layer |
| MATH-07 | Finite values | No NaN or infinity appears in task vectors, SVD outputs, coefficients, or merged weights |

Use explicit tolerances appropriate to dtype. Start with absolute and relative tolerances of 1e-5 for float32 unit tests and document any justified relaxation.

### 6.4 Baseline behavior tests

| Test ID | Test | Pass condition |
|---|---|---|
| BASE-01 | Pretrained evaluation | Completes on one batch and the full selected test split |
| BASE-02 | Individual expert evaluation | Each expert performs materially better than pretrained on its own task or discrepancy is investigated |
| BASE-03 | Task Arithmetic deterministic run | Repeated merge with the same inputs produces identical weights |
| BASE-04 | TIES retention | Configured trim ratio is applied and recorded |
| BASE-05 | Fair evaluation | Every method uses identical preprocessing, task heads, and test examples |

### 6.5 DOGE behavior tests

| Test ID | Test | Pass condition |
|---|---|---|
| DOGE-01 | Two-task smoke merge | DOGE completes for MNIST and SVHN |
| DOGE-02 | Optimization stability | Objective and parameter values remain finite for all iterations |
| DOGE-03 | Projection correctness | Update direction satisfies the implemented shared-subspace constraint |
| DOGE-04 | Output checkpoint | Merged checkpoint saves and reloads without changing predictions |
| DOGE-05 | Data-free boundary | No training or validation samples are accessed during DOGE merging |
| DOGE-06 | Evaluation separation | Test data is used only after the merged model has been produced |

### 6.6 End-to-end integration tests

| Test ID | Test | Pass condition |
|---|---|---|
| INT-01 | One-batch pipeline | Load, merge, evaluate, and save results succeeds |
| INT-02 | Pairwise pipeline | One complete pair runs for all three methods |
| INT-03 | Resume behavior | Interrupted experiment skips verified completed runs and resumes safely |
| INT-04 | Result schema | Every result contains all required metadata and metrics |
| INT-05 | Figure regeneration | Figures regenerate only from saved raw results |
| INT-06 | Notebook clean run | Notebook follows the documented sequence without hidden state |

### 6.7 Experimental validity checks

- Use the same dataset splits and preprocessing for every method.
- Do not tune a method using final test accuracy.
- Keep the base model and expert checkpoints fixed.
- Record all exclusions and failed runs.
- Use one canonical configuration per method before examining results.
- Report exact task sets, not only their size.
- If stochastic behavior exists, use at least three seeds or explain why the method is deterministic.
- Compare measured individual-expert accuracy against the paper or official repository; investigate large deviations before merging.
- Never manually copy final metrics into charts when they can be generated from structured results.

### 6.8 Final acceptance tests

- [x] One pairwise run can be reproduced from a fresh setup by following the README instructions.
- [x] All nine Tier 1 runs exist: P01, P03, and F01 times three methods.
- [x] Tier 2 similarity results exist for all six pairs.
- [x] All twelve Tier 3 runs exist: P02, P04, P05, and P06 times three methods.
- [x] Reference evaluations for pretrained and all four individual experts exist.
- [x] Every chart traces to a raw result artifact.
- [x] Mock results are absent from measured-results directories.
- [x] The submitted notebook and analysis tables use only verified measured results.

Presentation materials are not included in this repository.

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
[field mapping in INSTRUCTION.md §9](../INSTRUCTION.md#9-result-file-format).

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
[INSTRUCTION.md §6](../INSTRUCTION.md#6-re-run-the-experiments-gpu). The setup script creates the
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
