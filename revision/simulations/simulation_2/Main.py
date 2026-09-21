from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _legacy_wrapper import run_target


if __name__ == "__main__":
    run_target(Path(__file__).resolve().parents[1], "scripts/revision/run_sim2_ape_oob_comparison.py")
