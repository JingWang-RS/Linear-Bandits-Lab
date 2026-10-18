"""Read which environments, algorithms, rounds, and seeds an experiment uses."""

import copy
import json
from pathlib import Path

from algorithms import resolve_algorithms
from bandit_environments.config import fields, integer, resolve_environment

CONFIG_DIR = Path(__file__).parent / "configs"


def resolve_config(raw):
    config = copy.deepcopy(raw)
    fields(config, {"schema_version", "horizon", "seeds", "environment", "algorithms"}, "config")
    version = integer(config.get("schema_version", 2), "schema_version")
    if version != 2:
        raise ValueError("schema_version must be 2")
    horizon = integer(config.get("horizon"), "horizon")
    seeds = config.get("seeds")
    if not isinstance(seeds, list) or not seeds:
        raise ValueError("seeds must be a nonempty list")
    for seed in seeds:
        integer(seed, "seed", 0)
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be unique")
    raw_environment = config.get("environment")
    env = config["environment"] = resolve_environment(raw_environment)
    config["schema_version"] = 2
    config["algorithms"] = resolve_algorithms(config.get("algorithms"), env, horizon)
    return config


def config_path(source):
    path = Path(source).expanduser()
    if path.is_file():
        return path
    bundled = CONFIG_DIR / (str(source) if str(source).endswith(".json") else f"{source}.json")
    if bundled.is_file():
        return bundled
    raise ValueError(f"configuration not found: {source}")


def load_config(source, *, horizon=None, seeds=None):
    raw = json.loads(config_path(source).read_text())
    if not isinstance(raw, dict):
        raise ValueError("config must be an object")
    if horizon is not None:
        raw["horizon"] = horizon
    if seeds is not None:
        raw["seeds"] = seeds
    return resolve_config(raw)


def load_experiments(source):
    """Read one config, or a list of configs to run and plot together."""
    path = config_path(source).resolve()
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError("config must be an object")
    if "experiments" not in raw:
        config = resolve_config(raw)
        return {"title": path.stem, "experiments": [{
            "name": path.stem, "title": path.stem, "config": config,
            "algorithms": [entry["name"] for entry in config["algorithms"]],
            "runtime": False,
        }]}
    fields(raw, {"title", "experiments"}, "experiment collection")
    if not isinstance(raw.get("title"), str) or not raw["title"].strip():
        raise ValueError("experiment collection needs a title")
    if not isinstance(raw["experiments"], list) or not raw["experiments"]:
        raise ValueError("experiments must be a nonempty list")
    entries, names = [], set()
    for entry in raw["experiments"]:
        fields(entry, {"config", "title", "algorithms", "runtime"}, "experiment")
        if not isinstance(entry.get("config"), str):
            raise ValueError("each experiment needs a config filename")
        child = (path.parent / entry["config"]).resolve()
        name = child.stem
        if name in names or name in {".", ".."}:
            raise ValueError(f"duplicate or invalid experiment name: {name}")
        names.add(name)
        config = load_config(child)
        available = [algorithm["name"] for algorithm in config["algorithms"]]
        selected = entry.get("algorithms", available)
        if (not isinstance(selected, list) or not selected
                or any(not isinstance(item, str) or item not in available for item in selected)
                or len(set(selected)) != len(selected)):
            raise ValueError(f"{name}: plot algorithms must be distinct names from the config")
        title = entry.get("title", name)
        if not isinstance(title, str) or not title.strip():
            raise ValueError(f"{name}: title must be a nonempty string")
        runtime = entry.get("runtime", False)
        if not isinstance(runtime, bool):
            raise ValueError(f"{name}: runtime must be true or false")
        entries.append({"name": name, "title": title, "config": config,
                        "algorithms": selected, "runtime": runtime})
    return {"title": raw["title"], "experiments": entries}
