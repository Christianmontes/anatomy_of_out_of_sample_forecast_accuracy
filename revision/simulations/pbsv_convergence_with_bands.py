from pathlib import Path

from _legacy_wrapper import run_target


if __name__ == "__main__":
    run_target(Path(__file__).resolve().parent, "scripts/revision/run_gp_bsv_convergence_bands.py")
