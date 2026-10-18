"""Run the experiments in a config and save regret and requested runtime plots."""

import argparse
import json
import os
from pathlib import Path
import sys

# Support both the two root scripts and `python experiments/run.py --config ...`.
sys.dont_write_bytecode = True
# Set before importing NumPy/SciPy, including in spawned workers. Parallelism
# comes from independent trials; every trial uses one numerical-library thread.
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[variable] = "1"
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.config import load_experiments
from experiments.plot import collect_results, plot_results
from experiments.runner import run_experiment, validate_resume, write_json
from utils.experiment_io import ROOT, config_digest, metadata


def run(config_file, output_dir, *, workers=1, resume=False):
    """Run every config in the collection using the same experiment runner."""
    plan = load_experiments(config_file)
    output = Path(output_dir).resolve()
    data = output / "data"
    identity = {
        "plan": plan,
        "config_sha256": config_digest(plan),
        "code_sha256": metadata(plan)["code_sha256"],
    }
    manifest = data / "experiment.json"
    if data.exists():
        if not resume:
            raise ValueError(f"saved trials exist: {data}; use --resume or a new --output-dir")
        saved = json.loads(manifest.read_text())
        if any(saved.get(key) != value for key, value in identity.items()):
            raise ValueError("cannot resume: configuration or source changed; choose a new --output-dir")
        for entry in plan["experiments"]:
            validate_resume(entry["config"], data / entry["name"],
                            workers=1 if entry["runtime"] else workers)
    else:
        data.mkdir(parents=True)
    state = {**identity, "status": "running", "completed": []}
    write_json(manifest, state)
    try:
        for entry in plan["experiments"]:
            if metadata(plan)["code_sha256"] != identity["code_sha256"]:
                raise ValueError("source changed during the experiment")
            print(f"\n{entry['title']}", flush=True)
            trial_workers = 1 if entry["runtime"] else workers
            if entry["runtime"]:
                print("Runtime comparison: serial trials, one numerical thread.", flush=True)
            run_experiment(entry["config"], data / entry["name"], workers=trial_workers, resume=resume)
            if metadata(plan)["code_sha256"] != identity["code_sha256"]:
                raise ValueError("source changed during the experiment")
            state["completed"].append(entry["name"])
            write_json(manifest, state)
        collect_results(plan, data, output)
        state["status"] = "complete"
    except BaseException as exc:
        state.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error=str(exc))
        raise
    finally:
        write_json(manifest, state)
    return output / "regret.pdf"


def main(default_config=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=default_config, help="experiment JSON file")
    parser.add_argument("--workers", type=int, default=1,
                        help="parallel trials; runtime comparisons always run serially (default: 1)")
    parser.add_argument("--output-dir", type=Path, help="default: results/CONFIG_NAME")
    parser.add_argument("--resume", action="store_true", help="continue saved trials with the same code and config")
    parser.add_argument("--plot", action="store_true", help="only redraw the saved curves; do not run algorithms")
    args = parser.parse_args()
    if not args.config:
        parser.error("provide --config, or use one of the two exp_*.py scripts")
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.plot and args.resume:
        parser.error("choose --plot or --resume")
    output = args.output_dir or ROOT / "results" / Path(args.config).stem
    try:
        figure = (plot_results(output) if args.plot else
                  run(args.config, output, workers=args.workers, resume=args.resume))
        print(f"\nSaved: {figure}")
        if (Path(output) / "runtime.pdf").is_file():
            print(f"Saved: {Path(output).resolve() / 'runtime.pdf'}")
    except KeyboardInterrupt:
        print("Interrupted. Run the same command with --resume to continue.", file=sys.stderr)
        return 130
    except (ValueError, RuntimeError, OSError, ArithmeticError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
