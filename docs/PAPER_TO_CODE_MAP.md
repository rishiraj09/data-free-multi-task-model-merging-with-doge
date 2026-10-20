# DOGE paper-to-code map

This document maps the paper's equations and Algorithm 1 to the implementation used in the
experiments. Code references point to the official DOGE repository at
commit `ca0a0a1` in `third_party/DOGE`, downloaded by
[`scripts/setup_env.sh`](../scripts/setup_env.sh). Paper references are to
[Wei et al., ICML 2025](https://proceedings.mlr.press/v267/wei25k.html); the paper PDF is available
from that page and is not bundled with the submission. The similarity values in section 7 are
**paper-reported** and are only a sanity reference.

The experimental settings are defined in
[`configs/canonical_methods.yaml`](../configs/canonical_methods.yaml) (`config_id: canonical-v1`).
Configuration values are unchanged from the measured runs.
[Source text revisions](../results/manifests/source_text_revisions.json) document the exact
file-hash transitions for revised comments, docstrings and diagnostic wording; the original
hashes in the raw results are preserved. The [experimental protocol](EXPERIMENT_PROTOCOL.md)
defines the evaluation and verification criteria.

## 1. Symbol glossary

| Symbol | Meaning | In code (`DOGE_TA.py`) |
|---|---|---|
| θ0 | Pretrained CLIP ViT-B/32 vision encoder (`openai/clip-vit-base-patch32`) | `pretrained_model`, loaded as `_pretrained_` |
| θi | Expert fine-tuned on task i (`tanganke/clip-vit-base-patch32_<task>`) | `modelpool.load_model(name)` |
| n | Number of merged tasks (2 or 4 here) | `len(task_vectors)` |
| l | One linear layer (weight matrix) of the encoder | `layer_name` |
| τi | Task vector θi − θ0 | `task_vectors[i]` (filtered to linear weights) |
| τi^l | Task vector of task i at layer l (an m×d matrix) | `vector[layer_name]` |
| Δ | Modification vector, same shape as a task vector, shared by all tasks, starts at 0 | `self.delta` |
| η | Global scale of the merging coefficients (0.07 for vision) | `lamda` in the config |
| λi^l | Layer-wise coefficient η / ‖τi^l‖ | `self.lamdas[i][layer_name]` |
| Ui, Σi, Vi | SVD of τi^l | `u, s, v` |
| k | Per-task basis size: rank(τi^l) / n | `reduced_index_s` |
| Bi | First k left singular vectors of τi^l | columns of `sum_u` |
| S_share | Shared subspace: top left singular vectors of [B1, …, Bn] | `u_u[:, :rank/subspace]` |
| Proj_S(A) | B Bᵀ A, projection onto subspace S | `self.projection[layer_name] @ grad` |
| L(Δ) | Data-free objective, Eq. (5) | `taskvector_loss` |
| K | Fraction of each task vector kept by magnitude (30%) | `K`, `topk_values_mask` |
| θ* | Merged model θ0 + Σ λi(τi + Δ) | return value of `run` |
| λ (TA, TIES) | Single global scaling factor of the baselines | `scaling_factor` |

## 2. Where everything lives

All paths in the table below belong to the upstream DOGE checkout. Paths beginning with
`fusion_bench/`, `config/`, or `docs/` are relative to `third_party/DOGE/`, not to this project
root. They become available after setup; the upstream sources are also available at the
[pinned commit](https://github.com/WalkerWorldPeace/DOGE/tree/ca0a0a10c45bc0965f52f0815d6a6581ee103f85).
The `config/` directory upstream is distinct from this project's [`configs/`](../configs/).

| Item | File |
|---|---|
| DOGE TA algorithm | `third_party/DOGE/fusion_bench/method/DOGE_TA/DOGE_TA.py` |
| DOGE TA config | `third_party/DOGE/config/method/DOGE_TA/DOGE_TA.yaml` |
| Task Arithmetic | `fusion_bench/method/task_arithmetic/task_arithmetic.py`; config `config/method/task_arithmetic.yaml` |
| TIES-Merging | `fusion_bench/method/ties_merging/ties_merging.py` and `ties_merging_utils.py`; config `config/method/ties_merging.yaml` |
| Model pool (θ0, θi) | `fusion_bench/modelpool/clip_vision/modelpool.py` (`load_model`, line 78); configs `config/modelpool/CLIPVisionModelPool/`, e.g. `clip-vit-base-patch32_svhn_and_mnist.yaml` |
| Checkpoint ids | `config/model/clip-vit/clip-vit-base-patch32{,_mnist,_svhn,_gtsrb,_eurosat}.yaml` |
| Task pool (evaluation) | `fusion_bench/taskpool/clip_vision/taskpool.py`; config `config/taskpool/CLIPVisionModelTaskPool/_template.yaml` and `clip-vit-single-task_<task>.yaml` |
| Test datasets | `config/dataset/image_classification/test/{mnist,svhn,gtsrb,eurosat}.yaml` (`mnist`, `svhn`/`cropped_digits`, `tanganke/gtsrb`, `tanganke/eurosat`; split `test`) |
| Top-level program | `config/fabric_model_fusion.yaml` (`fusion_bench.programs.FabricModelFusionProgram`) |

The upstream repository has no ready-made four-task model pool or task pool for this task set,
nor a pair for MNIST-EuroSAT. This project builds them programmatically in
[`make_modelpool` and `make_taskpool`](../src/doge_exam/runner.py), using the task sets and
evaluation settings in [`configs/experiments.yaml`](../configs/experiments.yaml).
No additional model-pool or task-pool YAML files need to be created.

## 3. Algorithm 1 mapped to code (`DOGE_TA.py`)

| Algorithm 1 step | Code | Notes |
|---|---|---|
| Load θ0 and θi | `run`, lines 188-191 | Only model weights are loaded; no dataset is touched (data-free, DOGE-05) |
| τi ← θi − θ0 | `compute_task_vectors`, lines 82-105 | Keeps only keys with `encoder` and `weight` and without `layer_norm`: the 72 linear weight matrices (q, k, v, out_proj, fc1, fc2 × 12 blocks), 84,934,656 of 87,456,000 parameters |
| λi^l ← η / ‖τi^l‖ | `compute_layer_lamdas`, lines 286-294 (called at line 193) | Frobenius norm per layer and task |
| SVD of τi^l; Bi ← first k columns of Ui | `run`, lines 195-217 | k = floor(rank / n); `rank = min(m, d) = 768` for every ViT-B/32 linear layer, so k = 384 (n = 2) or 192 (n = 4) |
| S_share ← top singular vectors of [B1, …, Bn] | lines 218-223 | Second SVD of the stacked bases, keep floor(768 / `subspace`) = 128 columns; P^l = B Bᵀ (m × m) |
| L(Δ) | `taskvector_loss`, lines 107-130 | See discrepancy D2: inner products are taken per matrix row |
| ∇L ← ∇L − Proj_S(∇L) | `optimize_delta`, lines 159-164 | Removes the gradient component in S_share (left/output space) |
| Update Δ | `optimize_delta`, lines 133-171 | Adam, lr 1e-4, 400 iterations per layer, layers one after another (all hard-coded) |
| Keep top-K% of τi | `run`, lines 236-242; `topk_values_mask`, lines 296-314 | Mask from \|τi\| over the whole flattened 85M vector (not per layer), applied to τi + Δ |
| θ* ← θ0 + Σ λi(τi + Δ) | `run`, lines 243-270 | Non-linear parameters (embeddings, layer norms, biases) stay at θ0 |
| Evaluation | `taskpool.py`, `evaluate` line 272; heads at line 324 | The merged vision encoder is placed inside the pretrained `CLIPModel` (lines 296-299); zero-shot heads come from class names and templates through the pretrained text encoder, identical for every method (BASE-05) |

Baselines:

- **Task Arithmetic** (`task_arithmetic.py`, `run` lines 105-150): θ* = θ0 + λ Σ τi over **all**
  199 state-dict tensors (line 143).
- **TIES** (`ties_merging.py`, `run` line 70; `ties_merging_utils.py`): flatten all parameters
  (line 98), τ = θi − θ0 (line 102); trim each task to its top `threshold`% by magnitude
  (`topk_values_mask`, line 137); elect sign as the sign of the sum of trimmed vectors
  (`resolve_sign`, line 191); keep entries agreeing with the elected sign and combine them with
  `merge_func` (`disjoint_merge`, line 206); θ* = θ0 + λ τ_merged (line 110).

## 4. Official defaults

| Setting | Value | Source |
|---|---|---|
| DOGE iterations, lr, optimizer | 400 per layer, 1e-4, Adam | Hard-coded, `DOGE_TA.py` lines 143, 153; paper §5.1 and App. B |
| DOGE η | 0.07 | `DOGE_TA.yaml` `lamda`; paper §5.1 |
| DOGE per-task basis k | rank / n | code line 205; paper §5.1 |
| DOGE shared-subspace size | rank / 6 | `DOGE_TA.yaml` `subspace: 6`; paper App. B |
| DOGE magnitude mask | top 30% | `DOGE_TA.yaml` `K: 30`; paper §5.1 ("following TIES") |
| TA scaling factor | 0.5 in `task_arithmetic.yaml`; 0.3 in the FusionBench CLIP ViT-B/32 benchmark command | `config/method/task_arithmetic.yaml`; `docs/modelpool/clip_vit.md` line 348 |
| TIES scaling factor, threshold, merge | 0.5, 20, sum in `ties_merging.yaml`; 0.3, 20 in the benchmark command | `config/method/ties_merging.yaml`; `docs/modelpool/clip_vit.md` line 391 |
| Evaluation batch size, workers | 128, 8 | task-pool `_template.yaml` |

## 5. Paper-versus-code discrepancies

| ID | Paper | Code | Consequence |
|---|---|---|---|
| D1 | Algorithm 1 builds S_share from the first k columns, k = rank / n | Shared size is rank / 6 = 128, independent of n (App. B matches the code) | The shared subspace has the same size for 2 and 4 tasks; only the per-task bases shrink |
| D2 | Eq. (5): one squared inner product per task over the whole vector | `taskvector_loss` squares the inner product of each **row** of each layer matrix and sums them | The implemented objective is row-wise |
| D3 | DOGE retains the top 30% of parameters by magnitude | DOGE uses `K: 30`; the TIES baseline uses `threshold: 20` | The two methods use different mask fractions: 30% for DOGE, 20% for TIES |
| D4 | TIES paper merges agreeing entries by mean | FusionBench config uses `merge_func: sum` | All TIES experiments use the official `sum` setting |
| D5 | Algorithm 1 applies to all parameters | DOGE only changes the 72 linear weight matrices; other parameters stay at θ0, while TA and TIES merge all parameters | The parameter scope differs between DOGE and the baselines |
| D6 | — | Running time and memory are only printed (lines 168-170), not saved | [`src/doge_exam/runner.py`](../src/doge_exam/runner.py) records merge time and peak memory |
| D7 | §4.2: Δ is updated only in directions orthogonal to S_share | The gradient is projected, but Adam then rescales each coordinate, so the step is not orthogonal; measured on MNIST-SVHN, up to 27% of Δ's norm lies in S_share | The constraint holds for projected gradients, not for the accumulated Δ |

**Projection tolerance (DOGE-03).** The gradient projection check uses a relative tolerance of
1e-3 for ‖Pg‖/‖g‖, rather than 1e-5. The official implementation builds P = BBᵀ from a
float32 SVD; the largest absolute entry of PP − P was 1.9e-4, and the measured relative
projection residual was 8.1e-4. The tolerance accounts for this numerical error without
changing the official method. The check is implemented in
[`scripts/check_results.py`](../scripts/check_results.py).

## 6. Canonical settings (`canonical-v1`)

| Method | Settings | Why |
|---|---|---|
| Task Arithmetic | λ = 0.3 | Value used by FusionBench for the CLIP ViT-B/32 benchmark, and named in DOGE App. B as a good TA coefficient. The 0.5 in the yaml is a generic default not used in the benchmark docs |
| TIES | λ = 0.3, threshold 20 (top 20%), merge `sum` | FusionBench CLIP ViT-B/32 benchmark command; merge function is the config default |
| DOGE TA | η = 0.07, subspace 6, K 30, Adam lr 1e-4, 400 iterations | Official config and code, unchanged |

Evaluation controls:

- The same values are used for every task set (2-task and 4-task). No per-task-set tuning and no
  look at test or validation accuracy was involved.
- **Fixed-coefficient limitation:** TA and TIES normally tune λ on a validation set for each task set;
  a fixed λ = 0.3 taken from the 8-task benchmark may be sub-optimal for 2- and 4-task merges.
  DOGE's λi^l does not depend on n either, so the comparison is consistent.
- Seed 42 is used for all runs, as specified in
  [`configs/canonical_methods.yaml`](../configs/canonical_methods.yaml). TA and TIES are
  deterministic; DOGE is deterministic apart from GPU numerics.

## 7. Task-vector similarity definition (`encw-cos`)

- τi = θi − θ0 over the vision-model state-dict keys that contain `encoder` and `weight` and do
  not contain `layer_norm` (same filter as `DOGE_TA.compute_task_vectors`): 72 matrices,
  84,934,656 values.
- Flatten in sorted key order, concatenate, compute in float64:
  cos(τa, τb) = ⟨τa, τb⟩ / (‖τa‖ ‖τb‖).
- Results use `similarity__all-pairs__encw-cos__<UTC timestamp>.json` for all six pairs. The
  [submitted record](../results/raw/similarity__all-pairs__encw-cos__20261018T220404Z.json) is in
  [`results/raw/`](../results/raw/).
- Why this set: it is exactly the parameter space DOGE optimises, and it covers 97% of the
  vision encoder's parameters. Embeddings and layer norms are excluded because their scale differs.
- The paper does not state which parameters its Fig. 2(a) uses. Its **paper-reported** values
  for our pairs are: MNIST-SVHN 0.15, SVHN-GTSRB 0.07, MNIST-GTSRB 0.05, GTSRB-EuroSAT 0.03,
  SVHN-EuroSAT 0.02, MNIST-EuroSAT 0.01. Use these only to sanity-check our computation (same
  ordering, same small magnitude); never report them as our measurements.

## 8. Implementation and reproduction

- Checkpoint revisions and test-set fingerprints are pinned in
  [`configs/pins.yaml`](../configs/pins.yaml). The dependency pins include `datasets==3.2.0`
  for the legacy SVHN loader.
- [`src/doge_exam/runner.py`](../src/doge_exam/runner.py) builds the model and task pools, runs
  each method, and records the [protocol §7.2 fields](EXPERIMENT_PROTOCOL.md#72-minimum-result-fields),
  including merge time and peak memory (D6).
- DOGE memory: each layer's projection is at most 3072 × 3072 floats and the 85M-value mask
  vectors are built on CPU; the submitted experiments used a 24 GB RTX 4090.
- Follow [INSTRUCTION.md §6](../INSTRUCTION.md#6-re-run-the-experiments-gpu) to repeat the
  experiments using the included scripts.
