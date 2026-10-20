"""Run merges and evaluations with the official FusionBench components and save
result records with the fields of experimental protocol section 7.2.

Everything is built from configs/canonical_methods.yaml, configs/experiments.yaml and
configs/pins.yaml, the same way FusionBench's FabricModelFusionProgram does it: model pool
-> method.run(modelpool) -> CLIPVisionModelTaskPool.evaluate(merged_model).
Call configure_hf_cache(load_paths()) before importing this module.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import resource
import shutil
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import lightning as L
import torch
import yaml
from huggingface_hub import snapshot_download
from hydra.utils import instantiate as hydra_instantiate
from omegaconf import DictConfig, OmegaConf
from transformers import CLIPModel, CLIPProcessor, CLIPVisionModel

from fusion_bench.modelpool import CLIPVisionModelPool
from fusion_bench.taskpool import CLIPVisionModelTaskPool
from fusion_bench.utils import instantiate as fb_instantiate

from doge_exam.env_info import collect_environment
from doge_exam.metrics import summarize
from doge_exam.paths import PROJECT_ROOT, ProjectPaths
from doge_exam.probes import DataAccessProbe, DogeProbe

CONFIGS = PROJECT_ROOT / "configs"
DATASET_CONFIGS = PROJECT_ROOT / "third_party/DOGE/config/dataset/image_classification/test"
CONFIG_FILES = ("canonical_methods.yaml", "experiments.yaml", "pins.yaml")
CODE_FILES = (
    "src/doge_exam/runner.py",
    "src/doge_exam/metrics.py",
    "src/doge_exam/probes.py",
    "scripts/run_experiment.py",
)
WEIGHT_PATTERNS = ["*.json", "*.txt", "model.safetensors", "pytorch_model.bin"]


def load_config(name: str) -> dict:
    return yaml.safe_load((CONFIGS / name).read_text())


CANONICAL = load_config("canonical_methods.yaml")
EXPERIMENTS = load_config("experiments.yaml")
PINS = load_config("pins.yaml")


def _sha256_text(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def provenance() -> dict:
    return {
        "config_files": {f"configs/{n}": _sha256_text(CONFIGS / n) for n in CONFIG_FILES},
        "code_files": {n: _sha256_text(PROJECT_ROOT / n) for n in CODE_FILES},
        "git_commit": None,  # Original runs have no project commit; source and configuration hashes identify them.
    }


def local_model(repo_id: str) -> str:
    """Snapshot directory of the pinned revision (downloaded once into HF_HOME)."""
    return snapshot_download(repo_id, revision=PINS["models"][repo_id], allow_patterns=WEIGHT_PATTERNS)


def checkpoint_ids(tasks: list[str]) -> dict:
    ids = {"_pretrained_": CANONICAL["base_model"]} | {t: CANONICAL["experts"][t] for t in tasks}
    return {name: {"repo_id": rid, "revision": PINS["models"][rid]} for name, rid in ids.items()}


def make_fabric() -> L.Fabric:
    fabric = L.Fabric(accelerator="auto", devices=1, strategy="auto")
    fabric.launch()
    return fabric


def load_test_dataset(task: str):
    """Load a test set through the official FusionBench config and check it against the pins."""
    cfg = OmegaConf.load(DATASET_CONFIGS / f"{task}.yaml")[task]
    ds = hydra_instantiate(cfg)
    pin = PINS["datasets"][task]
    if len(ds) != pin["num_examples"] or ds._fingerprint != pin["fingerprint"]:
        raise RuntimeError(
            f"{task}: test set differs from pinned inventory (len {len(ds)} vs {pin['num_examples']}, "
            f"fingerprint {ds._fingerprint} vs {pin['fingerprint']})"
        )
    return ds


def make_taskpool(tasks: list[str], fabric: L.Fabric, fast_dev_run: bool) -> CLIPVisionModelTaskPool:
    base = local_model(CANONICAL["base_model"])
    processor = CLIPProcessor.from_pretrained(base)
    ev = EXPERIMENTS["evaluation"]
    taskpool = CLIPVisionModelTaskPool(
        {t: load_test_dataset(t) for t in tasks},
        processor=processor,
        data_processor=processor,
        clip_model=CLIPModel.from_pretrained(base),
        dataloader_kwargs={
            "batch_size": ev["batch_size"],
            "num_workers": ev["num_workers"],
            "pin_memory": True,
            "drop_last": False,
            "shuffle": False,
        },
        fast_dev_run=fast_dev_run,
    )
    taskpool._fabric_instance = fabric
    return taskpool


def make_modelpool(tasks: list[str]) -> CLIPVisionModelPool:
    models = {"_pretrained_": local_model(CANONICAL["base_model"])}
    models |= {t: local_model(CANONICAL["experts"][t]) for t in tasks}
    return CLIPVisionModelPool(
        DictConfig(models),
        processor=DictConfig(
            {
                "_target_": "transformers.CLIPProcessor.from_pretrained",
                "pretrained_model_name_or_path": models["_pretrained_"],
            }
        ),
    )


def make_method(method: str, fabric: L.Fabric):
    algorithm = fb_instantiate(DictConfig(CANONICAL["methods"][method]))
    if hasattr(algorithm, "_fabric_instance"):
        algorithm._fabric_instance = fabric
    return algorithm


def weights_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def evaluate(taskpool: CLIPVisionModelTaskPool, model: torch.nn.Module) -> tuple[dict, float]:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.time()
    report = taskpool.evaluate(model)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    tasks = [k for k in report if k not in ("model_info", "average")]
    per_task = {
        t: {
            "accuracy": 100 * report[t]["accuracy"],
            "loss": report[t]["loss"],
            "num_examples": len(taskpool.test_datasets[t]),
        }
        for t in tasks
    }
    return per_task, time.time() - start


def latest_expert_accuracy(paths: ProjectPaths, tasks: list[str]) -> tuple[dict, dict]:
    """Own-task accuracy of each expert from the newest complete reference record."""
    accuracy, sources = {}, {}
    for task in tasks:
        files = sorted(paths.raw_results.glob(f"ref__expert-{task}__*.json"))
        for path in reversed(files):
            record = json.loads(path.read_text())
            if record["status"] == "complete" and task in record["per_task"]:
                accuracy[task] = record["per_task"][task]["accuracy"]
                sources[task] = str(path.relative_to(paths.results_root.parent))
                break
    return accuracy, sources


def base_record(run_id: str, kind: str, method: str | None, tasks: list[str], fast_dev_run: bool) -> dict:
    env = collect_environment()
    return {
        "run_id": run_id,
        "kind": kind,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "fast_dev_run": fast_dev_run,
        "method": method,
        "task_set": tasks,
        "task_count": len(tasks),
        "seed": CANONICAL["seed"],
        "base_model": CANONICAL["base_model"],
        "config_id": CANONICAL["config_id"],
        "method_config": CANONICAL["methods"].get(method) if method else None,
        "evaluation_config": EXPERIMENTS["evaluation"],
        "device": env["device"],
        "precision": EXPERIMENTS["evaluation"]["precision"],
        "software": {"python": env["python_version"], **env["packages"], "doge_repo_commit": env["doge_repo_commit"]},
        "provenance": provenance(),
        "notes": [],
    }


def write_record(record: dict, paths: ProjectPaths) -> Path:
    folder = paths.results_root / "smoke" if record["fast_dev_run"] else paths.raw_results
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = folder / f"{record['run_id']}__{stamp}.json"
    if path.exists():
        raise FileExistsError(path)  # raw results are immutable
    path.write_text(json.dumps(record, indent=2, default=str))
    return path


def _run_safely(record: dict, paths: ProjectPaths, body) -> Path:
    try:
        body(record)
        record["status"] = "complete"
    except Exception:
        record["status"] = "failed"
        record["error"] = traceback.format_exc()
        print(write_record(record, paths))
        raise
    return write_record(record, paths)


def run_references(paths: ProjectPaths, fast_dev_run: bool = False) -> list[Path]:
    """Evaluate the pretrained model and every expert on all four test sets."""
    tasks = EXPERIMENTS["tasks"]
    L.seed_everything(CANONICAL["seed"])
    fabric = make_fabric()
    taskpool = make_taskpool(tasks, fabric, fast_dev_run)
    outputs = []
    for name in ["pretrained", *tasks]:
        repo_id = CANONICAL["base_model"] if name == "pretrained" else CANONICAL["experts"][name]
        label = "pretrained" if name == "pretrained" else f"expert-{name}"
        record = base_record(f"ref__{label}__{'-'.join(tasks)}", "reference", None, tasks, fast_dev_run)
        record["checkpoints"] = {"evaluated": {"repo_id": repo_id, "revision": PINS["models"][repo_id]}}

        def body(rec, repo_id=repo_id):
            model = CLIPVisionModel.from_pretrained(local_model(repo_id))
            rec["per_task"], rec["evaluation_time_seconds"] = evaluate(taskpool, model)
            rec["per_task_accuracy"] = {t: v["accuracy"] for t, v in rec["per_task"].items()}
            rec |= summarize(rec["per_task_accuracy"])
            del model

        path = _run_safely(record, paths, body)
        print(path)
        outputs.append(path)
    return outputs


def merge(experiment: str, method: str, fabric: L.Fabric) -> tuple[torch.nn.Module, dict]:
    tasks = EXPERIMENTS["experiments"][experiment]["tasks"]
    L.seed_everything(CANONICAL["seed"])
    modelpool = make_modelpool(tasks)
    algorithm = make_method(method, fabric)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    doge_probe = DogeProbe(algorithm) if hasattr(algorithm, "taskvector_loss") else None
    start = time.time()
    with DataAccessProbe() as data_probe, (doge_probe or contextlib.nullcontext()):
        merged = algorithm.run(modelpool)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    stats = {
        "merge_time_seconds": time.time() - start,
        "merge_finished_at": datetime.now(timezone.utc).isoformat(),
        "peak_memory_mb": torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else None,
        # ru_maxrss is in KiB on Linux; it is the process peak so far (model loading included)
        "peak_cpu_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "merged_weights_sha256": weights_sha256(merged),
        "merged_weights_finite": all(bool(torch.isfinite(p).all()) for p in merged.state_dict().values()),
        "data_access_during_merge": data_probe.summary(),
    }
    if doge_probe is not None:
        stats["doge_probe"] = doge_probe.summary()
        stats["probe_note"] = "objective recorded every iteration; projection checked at iterations 1, 200, 400"
    return merged, stats


def reload_check(model: torch.nn.Module, task: str, paths: ProjectPaths, run_id: str) -> dict:
    """DOGE-04: save the merged encoder, reload it and compare outputs on 64 test images."""
    folder = paths.cache_root / "tmp_reload" / run_id
    processor = CLIPProcessor.from_pretrained(local_model(CANONICAL["base_model"]))
    dataset = load_test_dataset(task)
    images = [dataset[i]["image"] for i in range(64)]
    pixels = processor(images=images, return_tensors="pt")["pixel_values"]
    device = next(model.parameters()).device
    try:
        model.save_pretrained(folder)
        reloaded = CLIPVisionModel.from_pretrained(folder).to(device).eval()
        model.eval()
        with torch.no_grad():
            a = model(pixel_values=pixels.to(device)).pooler_output
            b = reloaded(pixel_values=pixels.to(device)).pooler_output
        same_weights = weights_sha256(reloaded) == weights_sha256(model)
    finally:
        shutil.rmtree(folder, ignore_errors=True)
    max_diff = (a - b).abs().max().item()
    return {
        "task": task,
        "num_images": len(images),
        "max_abs_embedding_diff": max_diff,
        "weights_identical_after_reload": same_weights,
        "passed": same_weights and max_diff == 0.0,
    }


def run_id_for(experiment: str, method: str) -> str:
    spec = EXPERIMENTS["experiments"][experiment]
    return f"{spec['kind']}__{'-'.join(spec['tasks'])}__{method}__seed-{CANONICAL['seed']}"


def run_merge(experiment: str, method: str, paths: ProjectPaths, fast_dev_run: bool = False) -> Path:
    tasks = EXPERIMENTS["experiments"][experiment]["tasks"]
    record = base_record(run_id_for(experiment, method), "merge", method, tasks, fast_dev_run)
    record["experiment"] = experiment
    record["checkpoints"] = checkpoint_ids(tasks)

    def body(rec):
        fabric = make_fabric()
        merged, stats = merge(experiment, method, fabric)
        rec |= stats
        rec["peak_memory_note"] = "GPU peak during method.run only; Task Arithmetic and TIES merge on CPU"
        # DOGE-06: test data is loaded only after the merged model exists
        rec["test_data_first_loaded_at"] = datetime.now(timezone.utc).isoformat()
        taskpool = make_taskpool(tasks, fabric, fast_dev_run)
        rec["per_task"], rec["evaluation_time_seconds"] = evaluate(taskpool, merged)
        rec["reload_check"] = reload_check(merged, tasks[0], paths, rec["run_id"])
        rec["per_task_accuracy"] = {t: v["accuracy"] for t, v in rec["per_task"].items()}
        expert_acc, sources = latest_expert_accuracy(paths, tasks)
        rec["individual_expert_accuracy"] = expert_acc or None
        rec["reference_records"] = sources or None
        rec |= summarize(rec["per_task_accuracy"], expert_acc)
        if rec["normalized_retention"] is None:
            rec["notes"].append("normalized retention not computed: reference records missing")

    return _run_safely(record, paths, body)
