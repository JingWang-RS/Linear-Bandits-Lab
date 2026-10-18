"""Draw regret curves and runtime comparisons from saved experiment results."""

import hashlib
import json
from pathlib import Path

import numpy as np

from utils.experiment_io import atomic_file, config_digest


ALGORITHM_STYLES = {
    "weight-ucb": ("#e50000", "x"),
    "glb-weight-ucb": ("#e50000", "x"),
    "scb-weight-ucb": ("#e50000", "x"),
    "hvt-ucb": ("#e50000", "x"),
    "oful": ("#107ab0", "^"),
    "glb-static-ucb": ("#107ab0", "^"),
    "scb-static-ucb": ("#107ab0", "^"),
    "restart-ucb": ("#699d4c", "d"),
    "glb-restart-ucb": ("#699d4c", "d"),
    "scb-restart-ucb": ("#699d4c", "d"),
    "sliding-window-ucb": ("#8c6d31", "s"),
    "d-linucb": ("#cb416b", "o"),
    "bvd-glm-ucb": ("#cb416b", "o"),
    "master-oful": ("#f0833a", "H"),
    "master-glb": ("#f0833a", "H"),
    "master-scb": ("#f0833a", "H"),
    "crtm": ("#699d4c", "d"),
    "crmm": ("#887191", "h"),
    "heavy-oful": ("#f0833a", "H"),
    "glm-ucb-regularized": ("#107ab0", "^"),
    "ol2m": ("#699d4c", "d"),
    "mnl-ucb-local": ("#887191", "h"),
    "mlogb-ucb-variant": ("#f0833a", "H"),
}


def algorithm_style(algorithm):
    """Choose colors and markers by algorithm, independent of plot order or label."""
    identifier = algorithm["algorithm"]
    if identifier.startswith("master-") and algorithm["parameters"].get("n_init") == 0:
        return "#887191", "h"
    if identifier in ALGORITHM_STYLES:
        return ALGORITHM_STYLES[identifier]
    # New algorithms also get a repeatable style; red is reserved for our methods.
    colors = ["#107ab0", "#699d4c", "#887191", "#f0833a", "#cb416b", "#8c6d31"]
    markers = ["^", "d", "h", "H", "o", "s"]
    key = int.from_bytes(hashlib.sha256(identifier.encode()).digest()[:4], "big")
    return colors[key % len(colors)], markers[(key // len(colors)) % len(markers)]


def summarize(run_dir):
    """Compute each algorithm's mean regret and spread across random seeds."""
    run_dir = Path(run_dir)
    config = json.loads((run_dir / "config.json").read_text())
    summaries, curves = {}, {}
    for entry in config["algorithms"]:
        name = entry["name"]
        trials = []
        for seed in sorted(config["seeds"]):
            path = run_dir / "results" / name / f"seed-{seed}.npz"
            with np.load(path, allow_pickle=False) as result:
                curve = result["cumulative_regret"]
                if curve.shape != (config["horizon"],) or not np.isfinite(curve).all():
                    raise ValueError(f"invalid saved curve: {path}")
                trials.append(curve.copy())
        values = np.stack(trials)
        mean, std = values.mean(axis=0), values.std(axis=0, ddof=0)
        curves[name] = (mean, std)
        summaries[name] = {
            "trials": len(trials),
            "final_regret_mean": float(mean[-1]),
            "final_regret_std": float(std[-1]),
            "uncertainty": "one population standard deviation across saved trials",
        }
    return summaries, curves


def _settings(config):
    """Compare experimental inputs while allowing citation text to change."""
    from .config import resolve_config

    # Older saved configs may omit newly explicit defaults (e.g. centered noise).
    config = resolve_config(config)
    return {
        **{key: config[key] for key in ("horizon", "seeds", "environment")},
        "algorithms": [{key: entry[key] for key in ("name", "algorithm", "parameters")}
                       for entry in config["algorithms"]],
    }


def collect_results(plan, data_dir, output_dir):
    """Save one compact curve file with the settings needed to interpret it."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    arrays, experiments = {}, []
    for entry in plan["experiments"]:
        run = Path(data_dir) / entry["name"]
        if json.loads((run / "status.json").read_text()).get("status") != "complete":
            raise ValueError(f"experiment is incomplete: {run}")
        config = json.loads((run / "config.json").read_text())
        recorded = json.loads((run / "metadata.json").read_text())
        if config_digest(config) != recorded["config_sha256"]:
            raise ValueError(f"saved settings changed: {run}")
        if _settings(config) != _settings(entry["config"]):
            raise ValueError(f"saved settings differ from the selected experiment: {run}")
        for name in entry["algorithms"]:
            trials = []
            for seed in sorted(config["seeds"]):
                with np.load(run / "results" / name / f"seed-{seed}.npz", allow_pickle=False) as trial:
                    curve = trial["cumulative_regret"]
                    if curve.shape != (config["horizon"],) or not np.isfinite(curve).all():
                        raise ValueError(f"invalid saved curve: {run}/{name}/{seed}")
                    trials.append(curve.copy())
            arrays[f"{entry['name']}/{name}"] = np.stack(trials)
        saved_entry = {**entry, "config": config, "recorded_runtime": recorded}
        if entry.get("runtime", False):
            timings = json.loads((run / "timings.json").read_text())
            saved_entry["runtime_seconds"] = {
                name: [timings.get(f"{name}/seed-{seed}") for seed in sorted(config["seeds"])]
                for name in entry["algorithms"]
            }
            _runtime_values(saved_entry)
        experiments.append(saved_entry)
    details = {"schema_version": 1, "title": plan["title"], "experiments": experiments}
    arrays["metadata"] = np.asarray(json.dumps(details, ensure_ascii=False, allow_nan=False))
    with atomic_file(output / "curves.npz", "wb") as stream:
        np.savez_compressed(stream, **arrays)
    return plot_results(output)


def _runtime_values(entry):
    """Accept only complete, serial wall-clock measurements in seconds."""
    recorded = entry["recorded_runtime"]
    protocol = recorded.get("timing_protocol", {})
    if (recorded.get("trial_workers") != 1 or protocol.get("clock") != "time.perf_counter"
            or protocol.get("unit") != "seconds"):
        raise ValueError(f"{entry['name']}: runtime comparison needs recorded serial wall-clock timings")
    for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                     "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
        if recorded.get("thread_environment", {}).get(variable) != "1":
            raise ValueError(f"{entry['name']}: runtime comparison requires one numerical thread")
    values = []
    for name in entry["algorithms"]:
        seconds = entry.get("runtime_seconds", {}).get(name)
        if (not isinstance(seconds, list) or len(seconds) != len(entry["config"]["seeds"])
                or any(isinstance(value, bool) or not isinstance(value, (int, float))
                       or not np.isfinite(value) or value <= 0 for value in seconds)):
            raise ValueError(f"{entry['name']}/{name}: missing or invalid trial runtime")
        values.append(seconds)
    return np.asarray(values, dtype=float)


def _plot_runtime(output, details):
    """Plot the paper's mean time-per-trial bars on a logarithmic seconds axis."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    entries = [entry for entry in details["experiments"] if entry.get("runtime", False)]
    if not entries:
        return
    columns = min(3 if len(entries) >= 3 else 2, len(entries))
    rows = (len(entries) + columns - 1) // columns
    positions = [0, 1, 2, 4, 5] if len(entries) == 5 else list(range(len(entries)))
    figure = Figure(figsize=(6.3 * columns, 4.6 * rows), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(rows, columns, squeeze=False)
    for position, entry in zip(positions, entries, strict=True):
        axis = axes.flat[position]
        means = _runtime_values(entry).mean(axis=1)
        algorithms = {item["name"]: item for item in entry["config"]["algorithms"]}
        for index, (name, seconds) in enumerate(zip(entry["algorithms"], means, strict=True)):
            color, marker = algorithm_style(algorithms[name])
            axis.bar(index, seconds, color=color, alpha=0.7, edgecolor=color, width=0.65)
            axis.plot(index, seconds, marker=marker, color=color, markersize=7, linestyle="none")
            axis.annotate(f"{seconds:.2f}", (index, seconds), xytext=(0, 8),
                          textcoords="offset points", ha="center", fontsize=10)
        axis.set_xticks(np.arange(len(means)), entry["algorithms"], rotation=25, ha="right")
        axis.set_yscale("log")
        axis.set_ylim(means.min() * 0.45, means.max() * 2.2)
        config = entry["config"]
        axis.set(title=entry["title"] + f"\n{config['horizon']:,} rounds × {len(config['seeds'])} trials",
                 ylabel="Mean time per trial (seconds, log scale)")
        axis.grid(axis="y", alpha=0.15)
        axis.set_axisbelow(True)
    for position, axis in enumerate(axes.flat):
        if position not in positions:
            axis.set_visible(False)
    cpus = sorted({entry["recorded_runtime"].get("cpu", "CPU not recorded") for entry in entries})
    figure.suptitle(details["title"] + " — runtime\nSerial trials, one numerical thread · "
                    + "; ".join(cpus), fontsize=11)
    with atomic_file(output / "runtime.pdf", "wb") as stream:
        figure.savefig(stream, format="pdf")


def plot_results(output_dir):
    """Redraw regret and requested runtime figures from the single curves.npz file."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    output = Path(output_dir).resolve()
    with np.load(output / "curves.npz", allow_pickle=False) as saved:
        details = json.loads(str(saved["metadata"]))
        if details.get("schema_version") != 1 or not details.get("experiments"):
            raise ValueError("unsupported or empty curve file")
        experiments = details["experiments"]
        count = len(experiments)
        columns = min(3 if count >= 3 else 2, count)
        rows = (count + columns - 1) // columns
        positions = ([0, 1, 2, 4, 5] if count == 5 else
                     [0, 1, *range(3, 12)] if count == 11 else list(range(count)))
        figure = Figure(figsize=(6.3 * columns, 4.6 * rows), layout="constrained")
        FigureCanvasAgg(figure)
        axes = figure.subplots(rows, columns, squeeze=False)
        for index, entry in enumerate(experiments):
            axis = axes.flat[positions[index]]
            horizon, seeds = entry["config"]["horizon"], entry["config"]["seeds"]
            rounds = np.unique(np.r_[np.arange(0, horizon, 10), horizon - 1])
            algorithms = {item["name"]: item for item in entry["config"]["algorithms"]}
            for name in entry["algorithms"]:
                values = saved[f"{entry['name']}/{name}"]
                if (values.shape != (len(seeds), horizon) or values.dtype.kind not in "fiu"
                        or not np.isfinite(values).all()):
                    raise ValueError(f"invalid curves for {entry['name']}/{name}")
                mean = values.mean(axis=0)
                lower, upper = np.percentile(values, [5, 95], axis=0)
                color, marker = algorithm_style(algorithms[name])
                axis.plot(rounds, mean[rounds], label=name, color=color, marker=marker,
                          markevery=max(1, len(rounds) // 10), markersize=5, linewidth=1.6)
                axis.fill_between(rounds, lower[rounds], upper[rounds], color=color, alpha=0.15)
            axis.set(title=entry["title"], xlabel="Round", ylabel="Cumulative regret")
            axis.set_xlim(0, max(1, horizon - 1))
            axis.set_ylim(bottom=0)
            axis.legend(loc="upper left", fontsize=8)
            axis.grid(alpha=0.15)
        for i, axis in enumerate(axes.flat):
            if i not in positions:
                axis.set_visible(False)
        figure.suptitle(details["title"] + "\nMean and 5–95% trial percentiles", fontsize=13)
        with atomic_file(output / "regret.pdf", "wb") as stream:
            figure.savefig(stream, format="pdf")
    _plot_runtime(output, details)
    return output / "regret.pdf"
