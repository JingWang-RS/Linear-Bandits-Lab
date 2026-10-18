"""Run ICML 2025 main and appendix experiments, then draw regret and runtime figures."""

import sys

sys.dont_write_bytecode = True

from experiments.run import main


if __name__ == "__main__":
    raise SystemExit(main("heavy_tailed_LB"))
