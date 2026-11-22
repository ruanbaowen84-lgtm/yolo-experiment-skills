import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from experiment_config import load_plan
from experiment_runner import experiment_lock, run_experiment
from yolo_worker import train


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for split in ("train", "val", "test"):
            folder = self.root / "images" / split
            folder.mkdir(parents=True)
            (folder / "fixture.png").touch()
        (self.root / "data.yaml").write_text(
            "train: images/train\nval: images/val\ntest: images/test\nnames: [object]\n",
            encoding="utf-8",
        )
        for name in ("baseline", "candidate", "large"):
            (self.root / f"{name}.yaml").write_text("nc: 1\n", encoding="utf-8")
        self.config = {
            "data": "data.yaml",
            "output": "output",
            "baseline": "baseline.yaml",
            "candidates": [
                {"id": "candidate", "model": "candidate.yaml"},
                {"id": "large", "model": "large.yaml"},
            ],
            "train": {"epochs": 2, "device": "cpu"},
            "selection": {"max_candidates": 2},
            "final_test": True,
        }
        self.calls = []

    def plan(self):
        path = self.root / "experiment.yaml"
        path.write_text(yaml.safe_dump(self.config), encoding="utf-8")
        return load_plan(path)

    def worker(self, stage, case, plan, run_dir):
        case_id = case["id"]
        self.calls.append((stage, case_id))
        if stage == "train":
            weights = run_dir / "weights"
            weights.mkdir(exist_ok=True)
            (weights / "best.pt").write_bytes(b"fixture")
            return {"checkpoint": "weights/best.pt"}
        scores = {"baseline": (0.5, 0.3, 0.1), "candidate": (0.55, 0.35, 0.12), "large": (0.6, 0.4, 0.25)}
        map50, map5095, gflops = scores[case_id]
        return dict(map50=map50, map5095=map5095, gflops=gflops, parameters=42)

    def test_validation_selects_before_testing_and_repeated_run_reuses_evidence(self):
        plan = self.plan()
        output = self.root / "output"
        self.assertFalse(output.exists())
        result = run_experiment(plan, worker=self.worker)
        self.assertEqual(result["selected"], "candidate")
        self.assertEqual(self.calls, [
            ("train", "baseline"), ("val", "baseline"),
            ("train", "candidate"), ("val", "candidate"),
            ("train", "large"), ("val", "large"),
            ("test", "baseline"), ("test", "candidate"),
        ])
        self.calls.clear()
        self.assertEqual(run_experiment(plan, worker=self.worker), result)
        self.assertEqual(self.calls, [])
        self.assertEqual(json.loads((output / "metrics.json").read_text())["selected"], "candidate")
        self.assertNotIn("test", yaml.safe_load((output / "data.resolved.yaml").read_text()))
        self.assertIn("test", yaml.safe_load((output / "data.test.yaml").read_text()))

    def test_interrupted_train_retries_only_that_candidate(self):
        plan = self.plan()

        def interrupt(stage, case, config, run_dir):
            if stage == "train" and case["id"] == "candidate":
                weights = run_dir / "weights"
                weights.mkdir(exist_ok=True)
                (weights / "last.pt").write_bytes(b"checkpoint")
                raise KeyboardInterrupt
            return self.worker(stage, case, config, run_dir)

        with self.assertRaises(KeyboardInterrupt):
            run_experiment(plan, worker=interrupt)
        self.calls.clear()

        def resume(stage, case, config, run_dir):
            if stage == "train" and case["id"] == "candidate":
                self.assertEqual((run_dir / "weights" / "last.pt").read_bytes(), b"checkpoint")
            return self.worker(stage, case, config, run_dir)

        run_experiment(plan, worker=resume)
        self.assertEqual(self.calls[0], ("train", "candidate"))
        self.assertNotIn(("train", "baseline"), self.calls)

    def test_config_changes_are_rejected_in_existing_output(self):
        run_experiment(self.plan(), worker=self.worker)
        self.config["train"]["epochs"] = 3
        with self.assertRaisesRegex(ValueError, "configuration or inputs changed"):
            run_experiment(self.plan(), worker=self.worker)

    def test_changed_model_file_is_rejected_even_with_same_configuration(self):
        run_experiment(self.plan(), worker=self.worker)
        (self.root / "candidate.yaml").write_text("nc: 2\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "configuration or inputs changed"):
            run_experiment(self.plan(), worker=self.worker)

    def test_changed_image_or_label_rejects_a_mixed_dataset_run(self):
        folder = self.root / "labels" / "train"
        folder.mkdir(parents=True)
        label = folder / "fixture.txt"
        label.write_text("0 0.5 0.5 1 1\n", encoding="utf-8")
        original = self.plan()
        run_experiment(original, worker=self.worker)
        label.write_text("0 0.5 0.5 0.75 0.75\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "configuration or inputs changed"):
            run_experiment(self.plan(), worker=self.worker)
        labeled = self.plan()
        (self.root / "images" / "train" / "fixture.png").write_bytes(b"changed image")
        self.assertNotEqual(self.plan(), labeled)

    def test_missing_dataset_split_and_unsafe_ids_are_invalid(self):
        config = copy.deepcopy(self.config)
        self.config["candidates"][0]["id"] = "../escape"
        with self.assertRaisesRegex(ValueError, "candidate id"):
            self.plan()
        self.config = config
        (self.root / "images" / "val" / "fixture.png").unlink()
        with self.assertRaisesRegex(ValueError, "val.*no images"):
            self.plan()

    def test_held_out_list_cannot_reference_training_images(self):
        (self.root / "test.txt").write_text("./images/train/fixture.png\n", encoding="utf-8")
        data_file = self.root / "data.yaml"
        data = yaml.safe_load(data_file.read_text(encoding="utf-8"))
        data["test"] = "test.txt"
        data_file.write_text(yaml.safe_dump(data), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "data.test.*also appears"):
            self.plan()

    def test_recovers_when_final_epoch_was_saved_before_optimizer_stripping(self):
        plan = self.plan()
        run_dir = self.root / "baseline-run"
        weights = run_dir / "weights"
        weights.mkdir(parents=True)
        for name in ("last.pt", "best.pt"):
            (weights / name).touch()

        class Checkpoint:
            def __init__(self, path, task):
                self.ckpt = {"epoch": 1, "optimizer": {"state": {}}}

            def train(self, **kwargs):
                raise AssertionError("a completed epoch budget must not start training")

        result = train(Checkpoint, plan["cases"][0], plan, run_dir, self.root / "data.yaml")
        self.assertTrue(result["reused_finalized"])
        self.assertEqual(result["checkpoint"], "weights/best.pt")

    def test_fresh_initialization_keeps_only_boolean_pretrained_in_training_args(self):
        initial_weights = self.root / "initial.pt"
        initial_weights.touch()
        self.config["pretrained"] = "initial.pt"
        plan = self.plan()
        run_dir = self.root / "baseline-run"
        weights = run_dir / "weights"
        weights.mkdir(parents=True)
        calls = []

        class Model:
            def __init__(self, path, task):
                pass

            def load(self, checkpoint):
                calls.append(checkpoint)

            def train(self, **kwargs):
                calls.append(kwargs["pretrained"])
                (weights / "best.pt").touch()

        train(Model, plan["cases"][0], plan, run_dir, self.root / "data.yaml")
        self.assertEqual(calls, [str(initial_weights), True])

    def test_candidate_budget_and_duplicate_ids(self):
        self.config["selection"]["max_candidates"] = 1
        plan = self.plan()
        self.assertEqual([case["id"] for case in plan["cases"]], ["baseline", "candidate"])
        self.config["candidates"][1]["id"] = "candidate"
        with self.assertRaisesRegex(ValueError, "duplicate candidate id"):
            self.plan()

    def test_yolo_master_size_alias_resolves_and_preserves_name_for_model_loader(self):
        framework = self.root / "yolo"
        (framework / "ultralytics").mkdir(parents=True)
        (framework / "ultralytics" / "__init__.py").touch()
        original = self.root / "yolo-master.yaml"
        original.write_text("nc: 1\n", encoding="utf-8")
        self.config["framework_root"] = "yolo"
        self.config["baseline"] = "yolos-master.yaml"
        plan = self.plan()
        run_experiment(plan, worker=self.worker)
        staged = self.root / "output" / "baseline" / "model"
        self.assertEqual((staged / "yolos-master.yaml").read_bytes(), original.read_bytes())
        self.assertEqual((staged / "yolo-master.yaml").read_bytes(), original.read_bytes())

    def test_output_lock_releases_after_exit(self):
        output = self.root / "lock"
        output.mkdir()
        with experiment_lock(output):
            with self.assertRaisesRegex(ValueError, "already running"):
                with experiment_lock(output):
                    pass
        with experiment_lock(output):
            pass

    def test_a_live_worker_prevents_restart_after_parent_exits(self):
        plan = self.plan()
        output = self.root / "output"
        output.mkdir()
        with experiment_lock(output, ".worker.lock"):
            with self.assertRaisesRegex(ValueError, "already running"):
                run_experiment(plan, worker=self.worker)
        self.assertEqual(run_experiment(plan, worker=self.worker)["selected"], "candidate")


if __name__ == "__main__":
    unittest.main()
