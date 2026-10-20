"""Verify the pretrained and expert checkpoints and the four test datasets.

Runs CKPT-01 to CKPT-05, records sources, revisions, licences, sizes and checksums,
confirms the test splits and preprocessing, and writes
results/manifests/inventory__<timestamp>.json. Exits with status 1 if any check fails.

Usage: python scripts/verify_inventory.py
Downloads go to the configured cache root (HF_HOME defaults to <cache_root>/huggingface).
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone

from doge_exam.paths import PROJECT_ROOT, configure_hf_cache, load_paths

PATHS = load_paths()
PATHS.ensure()
configure_hf_cache(PATHS)  # must run before huggingface_hub reads HF_HOME

import yaml  # noqa: E402
from huggingface_hub import HfApi, try_to_load_from_cache  # noqa: E402
from hydra.utils import instantiate  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from transformers import CLIPProcessor, CLIPVisionModel  # noqa: E402

from doge_exam.env_info import collect_environment  # noqa: E402
from doge_exam.inventory import (  # noqa: E402
    compare_architecture,
    sha256_file,
    task_vector_roundtrip,
)

DOGE_CONFIG = PROJECT_ROOT / "third_party" / "DOGE" / "config"
CANONICAL = yaml.safe_load((PROJECT_ROOT / "configs" / "canonical_methods.yaml").read_text())
TASKS = ("mnist", "svhn", "gtsrb", "eurosat")  # Task order from experimental protocol section 5.1.
WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")
# Loading CLIPVisionModel from the full CLIP checkpoint drops the text tower; FusionBench's
# model pool loads the pretrained model the same way (modelpool.py, load_model).
EXPECTED_UNUSED_PREFIXES = ("text_model.", "logit_scale", "text_projection.", "visual_projection.")

api = HfApi()


def card_license(info) -> str | None:
    card = getattr(info, "card_data", None) or getattr(info, "cardData", None)
    if card is None:
        return None
    return card.get("license") if isinstance(card, dict) else getattr(card, "license", None)


def inspect_checkpoint(repo_id: str) -> tuple[dict, dict]:
    info = api.model_info(repo_id, files_metadata=True)
    model, loading = CLIPVisionModel.from_pretrained(
        repo_id, revision=info.sha, output_loading_info=True
    )
    weights = None
    for name in WEIGHT_FILES:
        path = try_to_load_from_cache(repo_id, name, revision=info.sha)
        if isinstance(path, str):
            sibling = next((s for s in info.siblings if s.rfilename == name), None)
            hub_sha = sibling.lfs.sha256 if sibling is not None and sibling.lfs else None
            local_sha = sha256_file(path)
            weights = {
                "file": name,
                "size_bytes": sibling.size if sibling is not None else None,
                "sha256": local_sha,
                "hub_lfs_sha256": hub_sha,
                "sha256_matches_hub": hub_sha == local_sha if hub_sha else None,
            }
            break
    record = {
        "repo_id": repo_id,
        "source": f"https://huggingface.co/{repo_id}",
        "revision": info.sha,
        "license": card_license(info),
        "weights": weights,
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "loading": {k: sorted(map(str, v)) for k, v in loading.items()},
    }
    return record, model.state_dict()


def check_loading(record: dict, pretrained: bool) -> dict:
    loading = record["loading"]
    unexpected = loading.get("unexpected_keys", [])
    if pretrained:
        undocumented = [k for k in unexpected if not k.startswith(EXPECTED_UNUSED_PREFIXES)]
    else:
        undocumented = unexpected
    weights_ok = record["weights"] is not None and record["weights"]["sha256_matches_hub"] is not False
    return {
        "missing_keys": loading.get("missing_keys", []),
        "mismatched_keys": loading.get("mismatched_keys", []),
        "unexpected_keys_count": len(unexpected),
        "undocumented_unexpected_keys": undocumented,
        "passed": not loading.get("missing_keys")
        and not loading.get("mismatched_keys")
        and not undocumented
        and weights_ok,
    }


def inspect_dataset(task: str, processor: CLIPProcessor) -> dict:
    from fusion_bench.tasks.clip_classification import get_classnames_and_templates

    cfg_path = DOGE_CONFIG / "dataset" / "image_classification" / "test" / f"{task}.yaml"
    cfg = OmegaConf.load(cfg_path)[task]
    repo_id = cfg.get("path") or cfg["_args_"][0]
    ds = instantiate(cfg)
    label_feature = ds.features["label"]
    labels = sorted(ds.unique("label"))
    classnames, templates = get_classnames_and_templates(task)
    pixel_values = processor(images=[ds[0]["image"]], return_tensors="pt")["pixel_values"]
    try:
        info = api.dataset_info(repo_id)
        hub = {"resolved_id": info.id, "revision": info.sha, "license": card_license(info)}
    except Exception as exc:  # record, do not hide, a metadata lookup failure
        hub = {"error": f"{type(exc).__name__}: {exc}"}
    num_classes = getattr(label_feature, "num_classes", None)
    passed = (
        len(ds) > 0
        and set(ds.column_names) >= {"image", "label"}
        and labels == list(range(len(classnames)))
        and (num_classes is None or num_classes == len(classnames))
        and tuple(pixel_values.shape) == (1, 3, 224, 224)
    )
    return {
        "official_config": str(cfg_path.relative_to(PROJECT_ROOT)),
        "load_dataset_args": OmegaConf.to_container(cfg),
        "hub": hub,
        "split": cfg["split"],
        "num_examples": len(ds),
        "columns": ds.column_names,
        "num_classes_feature": num_classes,
        "num_classnames": len(classnames),
        "num_templates": len(templates),
        "labels_present": len(labels),
        "fingerprint": ds._fingerprint,
        "processed_sample_shape": list(pixel_values.shape),
        "passed": bool(passed),
    }


def main() -> int:
    start = time.time()
    base_id = CANONICAL["base_model"]
    pretrained_record, pretrained_sd = inspect_checkpoint(base_id)
    ckpt01 = check_loading(pretrained_record, pretrained=True)

    experts, ckpt02, ckpt03, ckpt05 = {}, {}, {}, {}
    for task in TASKS:
        record, sd = inspect_checkpoint(CANONICAL["experts"][task])
        experts[task] = record
        ckpt02[task] = check_loading(record, pretrained=False)
        ckpt03[task] = compare_architecture(pretrained_sd, sd)
        ckpt05[task] = task_vector_roundtrip(pretrained_sd, sd) if ckpt03[task]["passed"] else {
            "passed": False,
            "reason": "architecture mismatch",
        }
        del sd

    all_records = [pretrained_record, *experts.values()]
    ckpt04 = {
        "passed": all(
            r["weights"] and r["weights"]["sha256"] and r["weights"]["size_bytes"] for r in all_records
        )
    }

    processor = CLIPProcessor.from_pretrained(base_id, revision=pretrained_record["revision"])
    image_proc = processor.image_processor.to_dict()
    preprocessing = {
        key: image_proc.get(key)
        for key in (
            "do_convert_rgb", "do_resize", "size", "resample", "do_center_crop",
            "crop_size", "do_rescale", "rescale_factor", "do_normalize", "image_mean", "image_std",
        )
    }
    datasets_info = {task: inspect_dataset(task, processor) for task in TASKS}

    checks = {
        "CKPT-01": ckpt01["passed"],
        "CKPT-02": all(v["passed"] for v in ckpt02.values()),
        "CKPT-03": all(v["passed"] for v in ckpt03.values()),
        "CKPT-04": ckpt04["passed"],
        "CKPT-05": all(v["passed"] for v in ckpt05.values()),
        "DATASETS": all(v["passed"] for v in datasets_info.values()),
    }
    record = {
        "kind": "inventory",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config_id": CANONICAL["config_id"],
        "environment": collect_environment(),
        "checks": checks,
        "all_passed": all(checks.values()),
        "pretrained": pretrained_record,
        "experts": experts,
        "details": {"CKPT-01": ckpt01, "CKPT-02": ckpt02, "CKPT-03": ckpt03, "CKPT-05": ckpt05},
        "preprocessing": {"processor": base_id, **preprocessing},
        "datasets": datasets_info,
        "elapsed_seconds": round(time.time() - start, 1),
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = PATHS.manifests / f"inventory__{stamp}.json"
    out.write_text(json.dumps(record, indent=2, default=str))
    for name, ok in checks.items():
        print(f"{name}: {'PASS' if ok else 'FAIL'}")
    print(out)
    return 0 if record["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
