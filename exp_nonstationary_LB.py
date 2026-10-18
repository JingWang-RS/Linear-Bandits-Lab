"""Run all TIT 2026 non-stationary experiments and plot regret and runtime."""

import sys

sys.dont_write_bytecode = True

from experiments.run import main


if __name__ == "__main__":
    raise SystemExit(main("nonstationary_LB"))
