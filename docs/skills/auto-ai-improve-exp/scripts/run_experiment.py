"""Command-line entry point for repeatable YOLO detection experiments."""

import argparse
import json
from pathlib import Path
import sys

import yaml

from experiment_config import load_plan
from experiment_runner import run_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="experiment YAML")
    parser.add_argument("--dry-run", action="store_true", help="print resolved configuration without training")
    args = parser.parse_args()
    try:
        plan = load_plan(args.config)
        if args.dry_run:
            print(json.dumps(plan, ensure_ascii=False, indent=2))
            return 0
        result = run_experiment(plan)
    except KeyboardInterrupt:
        print("Interrupted. Run the same command to use the saved checkpoint.", file=sys.stderr)
        return 130
    except (ValueError, RuntimeError, OSError, yaml.YAMLError) as error:
        print(f"Experiment error: {error}", file=sys.stderr)
        return 1
    print(f"Selected: {result['selected']}")
    print(f"Report: {Path(plan['output']) / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
