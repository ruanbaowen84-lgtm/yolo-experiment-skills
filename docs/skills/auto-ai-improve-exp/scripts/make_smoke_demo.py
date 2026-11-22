"""Generate a tiny original detection dataset and two model configurations."""

import argparse
from pathlib import Path
import random

from PIL import Image, ImageDraw
import yaml


def save_yaml(path, value):
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def make_demo(root, framework_root=None, epochs=1):
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise ValueError(f"Demo directory must be empty: {root}")
    rng = random.Random(1729)
    for split, count in (("train", 12), ("val", 4), ("test", 4)):
        images = root / "images" / split
        labels = root / "labels" / split
        images.mkdir(parents=True)
        labels.mkdir(parents=True)
        for index in range(count):
            x, y = rng.randint(5, 37), rng.randint(5, 37)
            width, height = rng.randint(14, 23), rng.randint(14, 23)
            image = Image.new("RGB", (64, 64), tuple(rng.randint(5, 50) for _ in range(3)))
            ImageDraw.Draw(image).rectangle((x, y, x + width - 1, y + height - 1), fill=(250, 160, 35))
            image.save(images / f"{index:03d}.png")
            box = ((x + width / 2) / 64, (y + height / 2) / 64, width / 64, height / 64)
            (labels / f"{index:03d}.txt").write_text(
                "0 " + " ".join(f"{coordinate:.6f}" for coordinate in box) + "\n", encoding="utf-8"
            )
    save_yaml(root / "data.yaml", {
        "path": ".", "train": "images/train", "val": "images/val", "test": "images/test", "names": ["rectangle"],
    })
    for name, kernel in (("baseline", 3), ("candidate", 5)):
        save_yaml(root / f"{name}.yaml", {
            "nc": 1,
            "backbone": [
                [-1, 1, "Conv", [8, 3, 2]], [-1, 1, "Conv", [8, 3, 2]],
                [-1, 1, "Conv", [16, kernel, 2]], [-1, 1, "Conv", [16, 3, 2]],
                [-1, 1, "Conv", [16, 3, 2]],
            ],
            "head": [[[2, 3, 4], 1, "Detect", ["nc"]]],
        })
    config = {
        "task": "detect", "data": "data.yaml", "output": "output", "baseline": "baseline.yaml",
        "candidates": [{"id": "kernel-5", "model": "candidate.yaml"}],
        "train": {"epochs": epochs, "imgsz": 64, "batch": 4, "device": "cpu", "workers": 0},
        "selection": {"max_candidates": 1, "min_map_sum_gain": 0.01, "max_gflops_growth": 0.5},
        "final_test": True,
    }
    if framework_root is not None:
        config["framework_root"] = str(framework_root.resolve())
    save_yaml(root / "experiment.yaml", config)
    return root / "experiment.yaml"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--framework-root", type=Path)
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    try:
        print(make_demo(args.output.resolve(), args.framework_root, args.epochs))
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
