"""Run one stage in a fresh process using the configured YOLO backend."""

import math
import os
from pathlib import Path
import sys

from experiment_runner import experiment_lock, read_json, write_json


def train(YOLO, case, plan, run_dir, data):
    last = run_dir / "weights" / "last.pt"
    best = run_dir / "weights" / "best.pt"
    resumed = last.is_file()
    if resumed:
        model = YOLO(str(last), task="detect")
        epoch = model.ckpt.get("epoch")
        if (epoch == -1 or epoch == plan["train"]["epochs"] - 1) and best.is_file():
            # A final checkpoint may exist before the framework strips its optimizer.
            YOLO(str(best), task="detect")
            return {"checkpoint": "weights/best.pt", "resumed": False, "reused_finalized": True}
        if type(epoch) is not int or not 0 <= epoch < plan["train"]["epochs"] - 1 or model.ckpt.get("optimizer") is None:
            raise RuntimeError(f"No resumable epoch or optimizer in {last}; choose a fresh output directory.")
        print(f"Resuming after checkpoint epoch {epoch + 1} from {last}")
        model.train(resume=True)
    else:
        model_file = run_dir / "model" / Path(case["model"]).name
        model = YOLO(str(model_file), task="detect")
        if plan["pretrained"]:
            model.load(plan["pretrained"])
        model.train(
            **plan["train"], data=str(data), split="val", project=str(run_dir.parent), name=run_dir.name,
            exist_ok=True, pretrained=bool(plan["pretrained"]), cache=False, plots=False,
            deterministic=True, save=True, val=True,
        )
    if not best.is_file():
        raise FileNotFoundError(f"Training produced no best checkpoint: {best}")
    return {"checkpoint": "weights/best.pt", "resumed": resumed}


def evaluate(YOLO, stage, plan, run_dir, data):
    from ultralytics.utils.torch_utils import get_flops, get_num_params

    model = YOLO(str(run_dir / "weights" / "best.pt"), task="detect")
    settings = plan["train"]
    metrics = model.val(
        data=str(data), split=stage, imgsz=settings["imgsz"], batch=settings["batch"],
        device=settings["device"], workers=settings["workers"], project=str(run_dir),
        name=f"eval-{stage}", exist_ok=True, plots=False, verbose=False, half=False,
    )
    model.model.fuse(verbose=False)
    result = {
        "map50": float(metrics.box.map50), "map5095": float(metrics.box.map),
        "parameters": int(get_num_params(model.model)),
        "gflops": float(get_flops(model.model, imgsz=settings["imgsz"])),
    }
    if not all(math.isfinite(value) for value in result.values()) or result["gflops"] <= 0:
        raise RuntimeError("Validation or fused GFLOPs profiling returned invalid metrics. Check the profiler dependencies and logs.")
    return result


def main(plan_file, case_id, stage):
    plan = read_json(plan_file)
    run_dir = plan_file.parent / case_id
    data = plan_file.parent / ("data.test.yaml" if stage == "test" else "data.resolved.yaml")
    case = next(case for case in plan["cases"] if case["id"] == case_id)
    framework_root = plan["framework_root"]
    if framework_root:
        sys.path.insert(0, framework_root)
    config_dir = plan_file.parent / ".yolo"
    config_dir.mkdir(exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(config_dir)
    os.environ["YOLO_AUTOINSTALL"] = "false"

    import torch
    import ultralytics
    from ultralytics import YOLO

    if framework_root and not Path(ultralytics.__file__).resolve().is_relative_to(framework_root):
        raise RuntimeError(f"Expected the local framework under {framework_root}, got {ultralytics.__file__}")
    runtime = {
        "python": sys.version.split()[0], "torch": torch.__version__,
        "ultralytics": ultralytics.__version__, "framework": str(Path(ultralytics.__file__).resolve()),
    }
    runtime_file = plan_file.parent / "runtime.json"
    if runtime_file.exists() and read_json(runtime_file) != runtime:
        raise RuntimeError("The Python or YOLO backend changed; use the original environment or a new output directory.")
    write_json(runtime_file, runtime)
    print(runtime, flush=True)
    device = plan["train"]["device"]
    if isinstance(device, int) and device >= torch.cuda.device_count():
        raise RuntimeError(f"CUDA device {device} is unavailable in the selected PyTorch environment.")
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable in the selected PyTorch environment.")

    if stage == "train":
        result = train(YOLO, case, plan, run_dir, data)
    else:
        result = evaluate(YOLO, stage, plan, run_dir, data)
    write_json(run_dir / f".{stage}.json", result)


if __name__ == "__main__":
    plan_file = Path(sys.argv[1])
    with experiment_lock(plan_file.parent, ".worker.lock"):
        main(plan_file, sys.argv[2], sys.argv[3])
