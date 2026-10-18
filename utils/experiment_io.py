"""Save experiment settings, source versions, and results so runs can be checked and resumed."""

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def config_digest(config):
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def timestamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")


def cpu_name():
    """Record the processor used for a runtime comparison."""
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                return line.partition(":")[2].strip()
    if platform.system() == "Darwin":
        result = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                capture_output=True, text=True, check=False)
        if result.returncode == 0:
            return result.stdout.strip()
    return platform.processor() or platform.machine()


def metadata(config):
    files = []
    for package in ("algorithms", "bandit_environments", "utils", "experiments"):
        files.extend(
            path for path in (ROOT / package).rglob("*") if path.suffix in {".py", ".json"}
        )
    files.extend(ROOT.glob("requirements*.txt"))
    files.append(ROOT / "setup.sh")
    files.extend(ROOT.glob("exp_*.py"))
    sources = {
        path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(set(files))
        if path.is_file()
    }
    versions = {}
    for name in ("numpy", "scipy", "matplotlib"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed"
    git = {"commit": None, "dirty": None}
    try:

        def command(*args):
            return subprocess.run(
                ["git", "-C", str(ROOT), *args],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )

        # Do not accidentally describe a parent repository containing this checkout.
        top = command("rev-parse", "--show-toplevel")
        if top.returncode == 0 and Path(top.stdout.strip()).resolve() == ROOT:
            commit = command("rev-parse", "--verify", "HEAD")
            status = command("status", "--porcelain", "--untracked-files=normal")
            git = {
                "commit": commit.stdout.strip() if commit.returncode == 0 else None,
                "dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
            }
    except (OSError, subprocess.TimeoutExpired):
        pass
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu": cpu_name(),
        "timing_protocol": {
            "clock": "time.perf_counter",
            "unit": "seconds",
            "scope": "algorithm initialization and complete bandit loop, including environment feedback and regret evaluation; excludes imports, environment construction, result-file writing, and plotting",
        },
        "versions": versions,
        "git": git,
        "code_sha256": config_digest(sources),
        "source_sha256": sources,
        "config_sha256": config_digest(config),
        "numpy_backend": np.show_config(mode="dicts"),
        "thread_environment": {
            key: os.environ.get(key)
            for key in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "BLIS_NUM_THREADS",
                "OMP_DYNAMIC",
                "MKL_DYNAMIC",
                "MKL_CBWR",
                "PYTHONHASHSEED",
            )
        },
        "environment_protocol": "each algorithm replays fresh environment streams for the trial seed",
        "algorithm_seed_protocol": "SeedSequence(seed,100,SHA256(instance name))",
    }


@contextmanager
def atomic_file(path, mode):
    """A final filename always denotes a completely written file."""
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode=mode, dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
            yield stream
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_json(path, value):
    with atomic_file(path, "w") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


