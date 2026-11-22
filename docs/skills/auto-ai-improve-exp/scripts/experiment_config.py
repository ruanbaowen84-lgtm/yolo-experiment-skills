"""Resolve experiment settings without importing PyTorch or starting training."""

import hashlib
import json
import math
import os
from pathlib import Path
import re

import yaml


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
ID_PATTERN = re.compile(r"[a-z0-9][a-z0-9-]{0,63}")
RESERVED_IDS = {"baseline", "con", "prn", "aux", "nul", "com1", "com2", "com3", "com4",
                "com5", "com6", "com7", "com8", "com9", "lpt1", "lpt2", "lpt3", "lpt4",
                "lpt5", "lpt6", "lpt7", "lpt8", "lpt9"}
TRAIN_DEFAULTS = {
    "epochs": 20, "imgsz": 640, "batch": 8, "device": "cpu", "workers": 0,
    "optimizer": "SGD", "patience": 0, "seed": 0, "amp": False,
}


def mapping(value, label, allowed=None):
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    if allowed is not None:
        unknown = set(value).difference(allowed)
        if unknown:
            raise ValueError(f"{label} has unknown keys: {', '.join(map(str, unknown))}")
    return value


def yaml_mapping(path):
    return mapping(yaml.safe_load(path.read_text(encoding="utf-8-sig")), str(path))


def resolve_path(base, value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a local path")
    return (base / Path(value).expanduser()).resolve()


def local_file(base, value, label, suffixes):
    path = resolve_path(base, value, label)
    if path.suffix.lower() not in suffixes or not path.is_file():
        raise ValueError(f"{label} must point to an existing {', '.join(suffixes)} file: {path}")
    return path


def model_files(base, value, label, local_framework):
    requested = resolve_path(base, value, label)
    stem = re.sub(r"^(yolo(?:e-)?v?\d+)[nslmx](?=-|$)", r"\1", requested.stem)
    if local_framework:
        stem = re.sub(r"^yolo[nslmx]-master(?=-|$)", "yolo-master", stem)
        stem = re.sub(r"^(yolo-master(?:-lite)?)-[nslmx]$", r"\1", stem)
    canonical = requested.with_name(stem + requested.suffix)
    source = requested if requested.is_file() else canonical
    if source.suffix.lower() not in (".yaml", ".yml") or not source.is_file():
        raise ValueError(f"{label} must point to a model YAML or a scale alias of one: {requested}")
    yaml_mapping(source)
    return requested, canonical, source


def number(value, label, minimum, maximum=None, integer=False):
    if type(value) not in ((int,) if integer else (int, float)):
        raise ValueError(f"{label} must be numeric")
    if not math.isfinite(value) or value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"{label} is out of range")
    return value


def training_settings(raw):
    settings = TRAIN_DEFAULTS | mapping(raw, "train", TRAIN_DEFAULTS)
    for key in ("epochs", "imgsz", "batch"):
        number(settings[key], f"train.{key}", 1, integer=True)
    for key in ("workers", "patience", "seed"):
        number(settings[key], f"train.{key}", 0, integer=True)
    if settings["imgsz"] < 32 or settings["imgsz"] % 32:
        raise ValueError("train.imgsz must be a positive multiple of 32")
    if settings["batch"] < 2:
        raise ValueError("train.batch must be at least 2")
    device = settings["device"]
    if isinstance(device, str) and device.isascii() and device.isdigit():
        device = int(device)
    if not (type(device) is int and device >= 0 or device in ("cpu", "mps")):
        raise ValueError("train.device must be cpu, mps, or a single non-negative GPU index")
    settings["device"] = device
    if not isinstance(settings["optimizer"], str) or not settings["optimizer"].strip():
        raise ValueError("train.optimizer must be a non-empty string")
    if type(settings["amp"]) is not bool:
        raise ValueError("train.amp must be true or false")
    return settings


def fingerprint(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def image_paths(path):
    if path.is_dir():
        return {
            item.resolve() for item in path.rglob("*")
            if item.suffix.lower() in IMAGE_SUFFIXES and item.is_file()
        }
    if path.is_file() and path.suffix.lower() == ".txt":
        entries = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        images = set()
        for entry in entries:
            item = path.parent / entry
            if not (Path(entry).is_absolute() or entry.startswith("./")):
                return set()
            if item.suffix.lower() not in IMAGE_SUFFIXES or not item.is_file():
                return set()
            images.add(item.resolve())
        return images
    return set()


def dataset_revision(images_by_split):
    """Track dataset membership and file revisions without hashing large image payloads."""
    digest = hashlib.sha256()
    image_dir = f"{os.sep}images{os.sep}"
    label_dir = f"{os.sep}labels{os.sep}"
    for image, split in sorted(images_by_split.items()):
        label = Path(label_dir.join(str(image).rsplit(image_dir, 1))).with_suffix(".txt")
        entries = []
        for path in (image, label):
            try:
                info = path.stat()
                entries.append((str(path), info.st_size, info.st_mtime_ns))
            except FileNotFoundError:
                entries.append((str(path), None, None))
        digest.update((json.dumps((split, entries)) + "\n").encode("utf-8"))
    return {"sha256": digest.hexdigest(), "images": len(images_by_split)}


def normalize_data(path, final_test):
    data = yaml_mapping(path)
    root = resolve_path(path.parent, data.get("path", "."), "data.path")
    names = data.get("names")
    if not (
        isinstance(names, list) and names and all(isinstance(name, str) and name for name in names)
        or isinstance(names, dict) and names and all(isinstance(name, str) and name for name in names.values())
        or names is None and type(data.get("nc")) is int and data["nc"] > 0
    ):
        raise ValueError("data.yaml needs a non-empty names list or mapping, or positive nc")
    if isinstance(names, dict) and {str(index) for index in names} != {str(index) for index in range(len(names))}:
        raise ValueError("data.names indices must start at 0 and be contiguous")
    if isinstance(names, dict):
        names = [value for _, value in sorted(names.items(), key=lambda item: int(item[0]))]
    if names is not None and "nc" in data and data["nc"] != len(names):
        raise ValueError("data.nc does not match data.names")
    if data.get("channels", 3) != 3:
        raise ValueError("only three-channel detection datasets are supported")
    resolved = {"path": str(root), "channels": 3}
    resolved.update({"names": names} if names is not None else {"nc": data["nc"]})
    inputs = {str(path): fingerprint(path)}
    seen_images = {}
    existing_splits = ("train", "val", "test") if final_test else ("train", "val")
    for split in existing_splits:
        value = data.get(split)
        entries = value if isinstance(value, list) else [value]
        paths = [resolve_path(root, entry, f"data.{split}") for entry in entries]
        if not paths:
            raise ValueError(f"data.{split} has no images or has an invalid image list")
        for item in paths:
            images = image_paths(item)
            if not images:
                raise ValueError(f"data.{split} has no images or has an invalid image list: {item}")
            for image in images:
                previous = seen_images.setdefault(image, split)
                if previous != split:
                    raise ValueError(f"data.{split} image also appears in {previous}: {image}")
            if item.is_file():
                inputs[str(item)] = fingerprint(item)
        resolved[split] = [str(item) for item in paths] if isinstance(value, list) else str(paths[0])
    return resolved, inputs, dataset_revision(seen_images)


def load_plan(config_path):
    config_path = Path(config_path).resolve()
    base = config_path.parent
    config = mapping(
        yaml_mapping(config_path), "experiment",
        {"task", "data", "output", "baseline", "candidates", "framework_root", "pretrained",
         "train", "selection", "final_test"},
    )
    if config.get("task", "detect") != "detect":
        raise ValueError("only task: detect is supported")
    final_test = config.get("final_test", False)
    if type(final_test) is not bool:
        raise ValueError("final_test must be true or false")
    dataset = local_file(base, config.get("data"), "data", (".yaml", ".yml"))
    data, inputs, revision = normalize_data(dataset, final_test)
    output = resolve_path(base, config.get("output", "experiment-output"), "output")
    framework_root = config.get("framework_root")
    if framework_root is not None:
        framework_root = resolve_path(base, framework_root, "framework_root")
        if not (framework_root / "ultralytics" / "__init__.py").is_file():
            raise ValueError(f"framework_root must contain an ultralytics source tree: {framework_root}")
    pretrained = config.get("pretrained")
    if pretrained is not None:
        pretrained = local_file(base, pretrained, "pretrained", (".pt",))
        inputs[str(pretrained)] = fingerprint(pretrained)
    train = training_settings(config.get("train", {}))
    selection = {"max_candidates": 5, "min_map_sum_gain": 0.01, "max_gflops_growth": 0.5}
    selection.update(mapping(config.get("selection", {}), "selection", selection))
    number(selection["max_candidates"], "selection.max_candidates", 1, integer=True)
    number(selection["min_map_sum_gain"], "selection.min_map_sum_gain", 0, maximum=2)
    number(selection["max_gflops_growth"], "selection.max_gflops_growth", 0)

    candidates = config.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("candidates must be a non-empty list")
    models = [("baseline", config.get("baseline"))]
    seen_ids = set(RESERVED_IDS)
    for item in candidates:
        candidate = mapping(item, "candidate", {"id", "model"})
        case_id = candidate.get("id")
        if not isinstance(case_id, str) or not ID_PATTERN.fullmatch(case_id):
            raise ValueError("candidate id must contain 1–64 lowercase letters, digits or hyphens")
        if case_id in seen_ids:
            raise ValueError(f"duplicate candidate id or reserved name: {case_id}")
        seen_ids.add(case_id)
        models.append((case_id, candidate.get("model")))

    cases = []
    for case_id, value in models[: 1 + selection["max_candidates"]]:
        requested, canonical, source = model_files(base, value, f"{case_id}.model", framework_root is not None)
        inputs[str(source)] = fingerprint(source)
        cases.append({
            "id": case_id, "model": str(requested),
            "model_source": str(source), "canonical_name": canonical.name,
        })

    return {
        "schema": 1, "data": data, "output": str(output), "cases": cases, "train": train,
        "selection": selection, "final_test": final_test, "inputs": inputs,
        "dataset_revision": revision,
        "framework_root": str(framework_root) if framework_root is not None else None,
        "pretrained": str(pretrained) if pretrained is not None else None,
        "omitted_candidates": [case_id for case_id, _ in models[len(cases):]],
    }


def plan_digest(plan):
    payload = json.dumps(plan, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
