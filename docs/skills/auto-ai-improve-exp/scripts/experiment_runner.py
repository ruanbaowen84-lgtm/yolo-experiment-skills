"""Run and persist the training, validation, and selection stages."""

from contextlib import contextmanager
import json
from pathlib import Path
import subprocess
import sys

import yaml

from experiment_config import plan_digest


def write_text(path, content):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def stage_model(case, run_dir):
    folder = run_dir / "model"
    folder.mkdir(exist_ok=True)
    source = Path(case["model_source"]).read_bytes()
    for name in {Path(case["model"]).name, case["canonical_name"]}:
        destination = folder / name
        temporary = destination.with_name(destination.name + ".tmp")
        temporary.write_bytes(source)
        temporary.replace(destination)


def write_json(path, value):
    write_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


@contextmanager
def experiment_lock(output, name=".lock"):
    with (output / name).open("a+b") as stream:
        if sys.platform == "win32":
            import msvcrt

            if stream.tell() == 0:
                stream.write(b"\0")
                stream.flush()
            stream.seek(0)
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as error:
                raise ValueError(f"experiment is already running: {output}") from error
        else:
            import fcntl

            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise ValueError(f"experiment is already running: {output}") from error
        yield


def run_worker(stage, case, plan, run_dir):
    result_path = run_dir / f".{stage}.json"
    if result_path.exists():
        return read_json(result_path)
    output = Path(plan["output"])
    log_path = run_dir / f"{stage}.log"
    command = [
        sys.executable, "-X", "utf8", "-u", str(Path(__file__).with_name("yolo_worker.py").resolve()),
        str(output / "plan.json"), case["id"], stage,
    ]
    print(f"{case['id']} / {stage}  [{log_path}]", flush=True)
    with log_path.open("ab") as log:
        process = subprocess.run(command, cwd=run_dir, stdout=log, stderr=subprocess.STDOUT)
    if process.returncode or not result_path.is_file():
        raise RuntimeError(f"{case['id']} / {stage} failed (exit {process.returncode}). See {log_path}")
    return read_json(result_path)


def run_stage(status_file, state, case, stage, plan, run_dir, worker):
    progress = state["cases"].setdefault(case["id"], {})
    if stage in progress:
        return progress[stage]
    state["active"] = {"case": case["id"], "stage": stage}
    state.pop("error", None)
    write_json(status_file, state)
    try:
        result = worker(stage, case, plan, run_dir)
    except (Exception, KeyboardInterrupt) as error:
        state["error"] = {"type": type(error).__name__, "message": str(error)}
        write_json(status_file, state)
        raise
    progress[stage] = result
    state["active"] = None
    write_json(status_file, state)
    return result


def selection_results(cases, baseline, selection):
    results = {}
    for case_id, metrics in cases.items():
        gain = metrics["map50"] + metrics["map5095"] - baseline["map50"] - baseline["map5095"]
        growth = metrics["gflops"] / baseline["gflops"] - 1
        eligible = (
            round(gain, 12) >= selection["min_map_sum_gain"]
            and round(growth, 12) < selection["max_gflops_growth"]
        )
        results[case_id] = {"map_sum_gain": gain, "gflops_growth": growth, "eligible": eligible}
    eligible_ids = [case_id for case_id, result in results.items() if result["eligible"]]
    winner = max(eligible_ids, key=lambda key: (
        cases[key]["map50"] + cases[key]["map5095"], -cases[key]["gflops"]
    ), default="baseline")
    return winner, results


def report_text(result, plan):
    lines = [
        "# YOLO detection experiment", "",
        f"Selected using validation: **{result['selected']}**", "",
        f"Seed: {plan['train']['seed']}. Epoch budget per run: {plan['train']['epochs']}.",
        f"Minimum validation mAP sum gain: {plan['selection']['min_map_sum_gain']:.4f}. "
        f"Fused GFLOPs growth limit: {plan['selection']['max_gflops_growth']:+.2%} (exclusive).",
        "Profile: fused model at the configured image size.", "",
    ]
    for split in ("val", "test"):
        if not result[split]:
            continue
        lines += [
            f"## {split}", "",
            "| Model | mAP50 | mAP50–95 | Fused params | Fused GFLOPs | Δ mAP sum | Δ GFLOPs | Selection |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
        for case_id, metrics in result[split].items():
            decision = result["candidates"].get(case_id) if split == "val" else None
            label = ("eligible" if decision["eligible"] else "below thresholds") if decision else "—"
            gain = f"{decision['map_sum_gain']:+.4f}" if decision else "—"
            growth = f"{decision['gflops_growth']:+.2%}" if decision else "—"
            lines.append(
                f"| {case_id} | {metrics['map50']:.4f} | {metrics['map5095']:.4f} | "
                f"{metrics['parameters']:,} | {metrics['gflops']:.6f} | {gain} | {growth} | {label} |"
            )
        lines.append("")
    if plan["omitted_candidates"]:
        lines += ["Outside candidate budget: " + ", ".join(plan["omitted_candidates"]), ""]
    lines += ["See metrics.json for exact values and state.json for completed stages.", ""]
    return "\n".join(lines)


def run_experiment(plan, worker=run_worker):
    output = Path(plan["output"])
    output.mkdir(parents=True, exist_ok=True)
    with experiment_lock(output):
        with experiment_lock(output, ".worker.lock"):
            pass
        return run_locked(plan, output, worker)


def run_locked(plan, output, worker):
    status_file = output / "state.json"
    digest = plan_digest(plan)
    state = {"plan_digest": digest, "active": None, "cases": {}}
    if status_file.exists():
        state = read_json(status_file)
        if state.get("plan_digest") != digest:
            raise ValueError("configuration or inputs changed, choose a new output directory")
    else:
        if any(path.name not in {".lock", ".worker.lock"} for path in output.iterdir()):
            raise ValueError("output must be an empty directory or an existing experiment")
        write_json(status_file, state)

    write_json(output / "plan.json", plan)
    training_data = {key: value for key, value in plan["data"].items() if key != "test"}
    write_text(output / "data.resolved.yaml", yaml.safe_dump(training_data, allow_unicode=True))
    if plan["final_test"]:
        write_text(output / "data.test.yaml", yaml.safe_dump(plan["data"], allow_unicode=True))
    validation = {}
    for case in plan["cases"]:
        case_id = case["id"]
        run_dir = output / case_id
        run_dir.mkdir(exist_ok=True)
        stage_model(case, run_dir)
        trained = run_stage(status_file, state, case, "train", plan, run_dir, worker)
        if not (run_dir / trained["checkpoint"]).is_file():
            raise ValueError(f"{case_id} trained checkpoint is missing: {run_dir / trained['checkpoint']}")
        validation[case_id] = run_stage(status_file, state, case, "val", plan, run_dir, worker)

    winner, candidates = selection_results(
        {key: metrics for key, metrics in validation.items() if key != "baseline"},
        validation["baseline"], plan["selection"],
    )
    result = {"selected": winner, "candidates": candidates, "val": validation, "test": {}}
    state["selected"] = winner
    write_json(status_file, state)
    if plan["final_test"]:
        for case in plan["cases"]:
            case_id = case["id"]
            if case_id in ("baseline", winner):
                result["test"][case_id] = run_stage(
                    status_file, state, case, "test", plan, output / case_id, worker
                )

    write_json(output / "metrics.json", result)
    write_text(output / "report.md", report_text(result, plan))
    state["complete"] = True
    write_json(status_file, state)
    return result
