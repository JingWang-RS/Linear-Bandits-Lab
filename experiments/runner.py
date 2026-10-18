"""Run each algorithm against the same bandit environment and save rewards and regret."""

import fcntl
import json
import multiprocessing
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from algorithms import create_algorithm
from bandit_environments import SyntheticEnvironment
from utils.seeding import algorithm_rng

from .config import resolve_config
from .plot import summarize
from utils.experiment_io import atomic_file as _atomic_file, config_digest, metadata, write_json

_STOP = None


def _initialize_worker(stop):
    global _STOP
    _STOP = stop


def _validate_trial(path, config):
    """Only complete, numerically consistent trials are eligible for reuse."""
    horizon, dimension = config["horizon"], config["environment"]["dimension"]
    required = {
        "observed_reward", "expected_reward", "optimal_expected_reward",
        "instant_regret", "cumulative_regret", "action_id", "outcome",
        "chosen_features", "noise_moment_root",
    }
    try:
        with np.load(path, allow_pickle=False) as arrays:
            if set(arrays.files) != required:
                raise ValueError("unexpected or missing arrays")
            for key in required:
                shape = (horizon, dimension) if key == "chosen_features" else (horizon,)
                values = arrays[key]
                if values.shape != shape or not np.isfinite(values).all():
                    raise ValueError(f"invalid shape or nonfinite values in {key}")
                if key in {"action_id", "outcome"} and values.dtype.kind not in "iu":
                    raise ValueError(f"{key} must contain integers")
            if np.any(arrays["action_id"] < 0) or np.any(arrays["outcome"] < -1):
                raise ValueError("invalid action or outcome identifier")
            regret = arrays["optimal_expected_reward"] - arrays["expected_reward"]
            if np.any(regret < 0) or not np.array_equal(regret, arrays["instant_regret"]):
                raise ValueError("pseudo-regret identity failed")
            if not np.array_equal(np.cumsum(regret), arrays["cumulative_regret"]):
                raise ValueError("cumulative pseudo-regret identity failed")
    except Exception as exc:
        raise ValueError(f"cannot resume invalid saved trial {path}: {exc}") from exc


def _validate_resume(output, config, current_metadata):
    saved_config = json.loads((output / "config.json").read_text())
    saved_metadata = json.loads((output / "metadata.json").read_text())
    if saved_config != config or saved_metadata.get("config_sha256") != config_digest(config):
        raise ValueError("cannot resume: saved configuration differs from the requested configuration")
    # Creation time and Git bookkeeping may change; execution inputs may not.
    for field in (
        "python", "platform", "machine", "versions", "numpy_backend", "thread_environment",
        "code_sha256", "source_sha256", "environment_protocol", "algorithm_seed_protocol",
        "trial_workers", "timing_protocol", "cpu",
    ):
        if field not in saved_metadata or saved_metadata[field] != current_metadata[field]:
            raise ValueError(f"cannot resume: {field} differs from the original run; use a new output directory")
    expected = {
        output / "results" / entry["name"] / f"seed-{seed}.npz"
        for entry in config["algorithms"] for seed in config["seeds"]
    }
    unexpected = set((output / "results").rglob("*.npz")) - expected
    if unexpected:
        raise ValueError(f"cannot resume: unexpected saved trials: {sorted(map(str, unexpected))}")
    for path in sorted(expected):
        if path.exists():
            _validate_trial(path, config)
    _read_timings(output / "timings.json")


def _read_timings(path):
    times = json.loads(path.read_text()) if path.exists() else {}
    if not isinstance(times, dict) or any(
        value is not None and (
            isinstance(value, bool) or not isinstance(value, (int, float))
            or not np.isfinite(value) or value < 0
        )
        for value in times.values()
    ):
        raise ValueError(f"invalid saved timings: {path}")
    return times


def validate_resume(config, output_dir, *, workers=1):
    """Read-only preflight for suites; the runner rechecks under its write lock."""
    config = resolve_config(config)
    output = Path(output_dir).resolve()
    if not output.exists():
        return
    if not output.is_dir():
        raise ValueError(f"output is not a directory: {output}")
    if not any(path.name != ".run.lock" for path in output.iterdir()):
        return
    if not all((output / name).is_file() for name in ("config.json", "metadata.json")):
        raise ValueError(f"cannot resume: missing configuration or metadata in {output}")
    _validate_resume(output, config, {**metadata(config), "trial_workers": workers})


def _run_trial(config, entry, seed, destination):
    """Write one independent trial; return timing or precise failure details."""
    name = entry["name"]
    round_index = None
    try:
        if _STOP is not None and _STOP.is_set():
            return None
        env = SyntheticEnvironment(config["environment"], config["horizon"], seed)
        # Measure initialization and the complete interaction loop. Imports,
        # environment construction, file writing, and plotting are excluded.
        started = time.perf_counter()
        policy = create_algorithm(
            entry, config["environment"], config["horizon"], algorithm_rng(seed, name)
        )
        horizon = config["horizon"]
        arrays = {
            key: np.empty(horizon)
            for key in [
                "observed_reward",
                "expected_reward",
                "optimal_expected_reward",
                "instant_regret",
                "cumulative_regret",
            ]
        }
        arrays["action_id"] = np.empty(horizon, dtype=np.int64)
        arrays["outcome"] = np.full(horizon, -1, dtype=np.int64)
        arrays["chosen_features"] = np.empty((horizon, config["environment"]["dimension"]))
        arrays["noise_moment_root"] = np.full(horizon, -1.0)
        cumulative = 0.0
        for round_index in range(horizon):
            if _STOP is not None and _STOP.is_set():
                return None
            features, ids = env.context(round_index)
            index = policy.select(features, ids)
            if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
                raise ValueError("policy must return an integer candidate index")
            if not 0 <= index < len(features):
                raise ValueError("policy selected an unavailable action")
            means = env.expected_rewards(round_index, features)
            feedback = env.feedback(round_index, features, index)
            policy.update(features[index], feedback)
            gap = float(np.max(means) - means[index])
            cumulative += gap
            arrays["observed_reward"][round_index] = feedback.reward
            arrays["expected_reward"][round_index] = means[index]
            arrays["optimal_expected_reward"][round_index] = np.max(means)
            arrays["instant_regret"][round_index] = gap
            arrays["cumulative_regret"][round_index] = cumulative
            arrays["action_id"][round_index] = ids[index]
            arrays["chosen_features"][round_index] = features[index]
            if feedback.outcome is not None:
                arrays["outcome"][round_index] = feedback.outcome
            if feedback.noise_moment_root is not None:
                arrays["noise_moment_root"][round_index] = feedback.noise_moment_root
        if any(not np.isfinite(v).all() for v in arrays.values()):
            raise ArithmeticError("trial produced nonfinite results")
        elapsed = time.perf_counter() - started
        with _atomic_file(Path(destination) / f"seed-{seed}.npz", "wb") as stream:
            np.savez_compressed(stream, **arrays)
        return {"elapsed": elapsed}
    except BaseException as exc:
        return {
            "status": "interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed",
            "algorithm": name,
            "seed": seed,
            "round": round_index,
            "error_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }


def run_experiment(config, output_dir, *, workers=1, resume=False):
    """Run or safely resume complete trials without changing their random streams.

    Resume requires identical configuration, source files and numerical runtime.
    An unfinished trial starts over; completed trials are checked and retained.
    """
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    config = resolve_config(config)
    output = Path(output_dir).resolve()
    if not isinstance(resume, bool):
        raise ValueError("resume must be a boolean")
    output.mkdir(parents=True, exist_ok=resume)
    with (output / ".run.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"another process is already running this experiment: {output}") from exc
        try:
            return _run_experiment(config, output, workers=workers, resume=resume)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _run_experiment(config, output, *, workers, resume):
    current_metadata = {**metadata(config), "trial_workers": workers}
    existing = (output / "config.json").exists() or (output / "metadata.json").exists()
    if resume and existing:
        _validate_resume(output, config, current_metadata)
    else:
        # --resume can also start a not-yet-created member of an experiment suite.
        if any(path.name != ".run.lock" for path in output.iterdir()):
            raise ValueError(f"refusing to initialize a nonempty output directory: {output}")
        write_json(output / "config.json", config)
        write_json(output / "metadata.json", current_metadata)
    write_json(output / "status.json", {"status": "running"})
    name = seed = None
    timing_path = output / "timings.json"
    times = {}
    executor = stop = failure = None
    try:
        if resume:
            times = _read_timings(timing_path)
        tasks = []
        trial_keys = []
        for entry in config["algorithms"]:
            destination = output / "results" / entry["name"]
            destination.mkdir(parents=True, exist_ok=resume)
            for seed in config["seeds"]:
                key = f"{entry['name']}/seed-{seed}"
                trial_keys.append(key)
                if (resume and (destination / f"seed-{seed}.npz").is_file()
                        and times.get(key) is not None):
                    continue
                # A worker may save its trial just before interruption. Rerun
                # that trial if its timing was not saved; never invent a time.
                times.pop(key, None)
                tasks.append((entry, seed, destination))
        if resume:
            print(f"Resume: {len(trial_keys) - len(tasks)} validated trials retained; {len(tasks)} remaining", flush=True)
        write_json(timing_path, times)
        if workers == 1 or not tasks:
            completed = (
                (entry["name"], trial_seed, _run_trial(config, entry, trial_seed, destination))
                for entry, trial_seed, destination in tasks
            )
        else:
            context = multiprocessing.get_context("spawn")
            stop = context.Event()
            executor = ProcessPoolExecutor(
                max_workers=min(workers, len(tasks)),
                mp_context=context,
                initializer=_initialize_worker,
                initargs=(stop,),
            )
            futures = {
                executor.submit(_run_trial, config, entry, trial_seed, destination): (
                    entry["name"], trial_seed
                )
                for entry, trial_seed, destination in tasks
            }

            def parallel_results():
                for future in as_completed(futures):
                    trial_name, trial_seed = futures[future]
                    try:
                        result = future.result()
                    except Exception as exc:
                        result = {
                            "status": "failed",
                            "algorithm": trial_name,
                            "seed": trial_seed,
                            "round": None,
                            "error_type": type(exc).__name__,
                            "message": str(exc),
                            "traceback": traceback.format_exc(),
                        }
                    yield trial_name, trial_seed, result

            completed = parallel_results()
        for name, seed, result in completed:
            if result is None:
                raise RuntimeError("trial cancelled before completion")
            if result.get("status") == "interrupted":
                failure = result
                raise KeyboardInterrupt(result["message"])
            if result.get("status") == "failed":
                failure = result
                raise RuntimeError(result["message"])
            times[f"{name}/seed-{seed}"] = result["elapsed"]
            write_json(timing_path, times)
            print(f"{name}: seed {seed} complete ({config['horizon']} rounds)", flush=True)
        # Preserve the configured ordering even when trials finish out of order.
        times = {key: times[key] for key in trial_keys}
        write_json(output / "timings.json", times)
        write_json(output / "status.json", {"status": "results-complete"})
        summary, _ = summarize(output)
        write_json(output / "summary.json", summary)
        write_json(output / "status.json", {"status": "complete"})
    except BaseException as exc:
        interrupted = isinstance(exc, (KeyboardInterrupt, SystemExit))
        failure = failure or {
            "status": "interrupted" if interrupted else "failed",
            "algorithm": name,
            "seed": seed,
            "round": None,
            "error_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        write_json(output / "status.json", failure)
        if interrupted:
            raise
        raise RuntimeError(
            f"experiment failed: algorithm={failure['algorithm']}, seed={failure['seed']}, "
            f"round={failure['round']}: {failure['message']}"
        ) from exc
    finally:
        if executor is not None:
            stop.set()
            executor.shutdown(wait=True, cancel_futures=True)
    return output
