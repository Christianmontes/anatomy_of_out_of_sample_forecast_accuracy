"""
MAS from the simulation in-sample vs out-of-sample GPBSV (referee response).

Implements revision/submission_2/planning/sim_mas_is_vs_oos_gpbsv_decisions.md.

It does NOT recompute any GPBSV. It loads the already-computed simulation
results and computes Model Accordance Scores between the in-sample and
out-of-sample GPBSV (and two importance->OOS bridges), reusing the *paper's*
MAS functions from mas_referee_response/mas_referee_variants.py so the metric is
identical to the empirical referee table.

Inputs (existing):
  outputs/insample_gpbsv/<scenario>/is_vs_oos_gpbsv_per_seed.csv
      -> is_gpbsv_mse, oos_gpbsv_mse, is_gpbsv_rmse, oos_gpbsv_rmse (per seed/model/feature)
  outputs/sim1_master_out/<scenario>/table_{model}_seed{seed}.csv
      -> perm_import_train, oShapley_VI (per feature)

Comparisons (right side always signed-rank(OOS GPBSV); MSE headline + RMSE check):
  a  concordance:   signed-rank(IS GPBSV)        vs signed(OOS GPBSV); w = |IS GPBSV|
  b  robustness:    ordinary-rank(|IS GPBSV|)    vs signed(OOS GPBSV); w = |IS GPBSV|
  c1 in-sample bridge: ordinary-rank(perm_import_train) vs signed(OOS GPBSV); w = perm_import_train
  c2 OOS-Shapley bridge: ordinary-rank(oShapley_VI)     vs signed(OOS GPBSV); w = oShapley_VI

Reporting (P=4 is coarse): headline = mean +/- sd of the per-seed MAS over the
25 seeds; supplements = MAS on the seed-averaged feature GPBSVs (p=4, with MC
p-values at alpha 0.5 and 2/3), Spearman over the 25x4 cells (descriptive), and
sign/quadrant concordance shares.

Usage:  python run_sim1_mas_is_vs_oos_gpbsv.py
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
SIM_ROOT = SCRIPT_DIR.parents[1]              # revision/simulations
REPO_ROOT = SIM_ROOT.parents[1]               # anatomy_replication
MAS_DIR = REPO_ROOT / "revision" / "mas_referee_response"
if str(MAS_DIR) not in sys.path:
    sys.path.insert(0, str(MAS_DIR))

# Reuse the paper's MAS primitives verbatim (NOT compute_mas/prepare_series,
# which carry empirical labels and drop a base_contribution column we do not have).
from mas_referee_variants import (  # noqa: E402
    rank_importance,
    signed_rank_lower_loss_is_better,
    expected_signed_msdr_weighted,
    monte_carlo_p_value,
)

INSAMPLE_DIR = SIM_ROOT / "outputs" / "insample_gpbsv"
MASTER_DIR = SIM_ROOT / "outputs" / "sim1_master_out"
OUT_DIR = SIM_ROOT / "outputs" / "sim_mas_is_vs_oos_gpbsv"

SCENARIOS = [
    "break_b08_expanding",
    "persistent_rho099_rolling20",
    "proxy_break_reverse_noisyx1_expanding",
]
MODELS = ["ols", "enet", "rf"]
FEATURES = ["x1", "x2", "x3", "x4"]
SEEDS = list(range(100, 125))

COMPARISONS = ["a", "b", "c1", "c2"]
COMP_LABEL = {
    "a": "signed IS-GPBSV vs signed OOS-GPBSV (concordance)",
    "b": "|IS-GPBSV| vs signed OOS-GPBSV (robustness)",
    "c1": "perm_import_train vs signed OOS-GPBSV (in-sample bridge)",
    "c2": "oShapley_VI vs signed OOS-GPBSV (OOS-Shapley bridge)",
}
LOSSES = ["mse", "rmse"]

N_SIMS = 100_000
CHUNK = 50_000
P_ALPHAS = (0.5, 2.0 / 3.0)


def stable_seed(*parts: object) -> int:
    payload = "|".join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def load_insample(scenario: str) -> pd.DataFrame:
    p = INSAMPLE_DIR / scenario / "is_vs_oos_gpbsv_per_seed.csv"
    if not p.exists():
        raise FileNotFoundError(
            f"Missing IS/OOS GPBSV for scenario '{scenario}': {p}. "
            f"This script only supports scenarios with in-sample GPBSV."
        )
    df = pd.read_csv(p)
    need = {"model", "seed", "feature", "is_gpbsv_mse", "oos_gpbsv_mse",
            "is_gpbsv_rmse", "oos_gpbsv_rmse"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"{p} missing columns: {sorted(missing)}")
    return df


def load_importance(scenario: str, model: str, seed: int) -> pd.DataFrame:
    p = MASTER_DIR / scenario / f"table_{model}_seed{seed}.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing importance table: {p}")
    t = pd.read_csv(p, index_col=0)
    for col in ("perm_import_train", "oShapley_VI"):
        if col not in t.columns:
            raise ValueError(f"{p} has no '{col}' column (cols: {list(t.columns)})")
    return t


def get_vectors(sub: pd.DataFrame, imp: pd.DataFrame, comp: str, loss: str):
    """Return (left, right_oos, weights, left_rank_type) as length-4 arrays in FEATURES order."""
    oos = sub[f"oos_gpbsv_{loss}"].to_numpy(dtype=float)
    is_g = sub[f"is_gpbsv_{loss}"].to_numpy(dtype=float)
    if comp == "a":
        return is_g, oos, np.abs(is_g), "signed"
    if comp == "b":
        return np.abs(is_g), oos, np.abs(is_g), "ordinary"
    if comp == "c1":
        v = imp["perm_import_train"].to_numpy(dtype=float)
        return v, oos, np.clip(v, 0.0, None), "ordinary"   # importance weights are non-negative
    if comp == "c2":
        v = imp["oShapley_VI"].to_numpy(dtype=float)
        return v, oos, np.clip(v, 0.0, None), "ordinary"
    raise ValueError(comp)


def one_mas(left, right, weights, left_rank_type):
    """Compute MAS (value only) using the paper's signed-null normalizer (alpha 0.5)."""
    left_s = pd.Series(np.asarray(left, dtype=float))
    right_s = pd.Series(np.asarray(right, dtype=float))
    w = np.asarray(weights, dtype=float)
    ranks_left = (
        signed_rank_lower_loss_is_better(left_s)
        if left_rank_type == "signed"
        else rank_importance(left_s)
    ).to_numpy(dtype=float)
    ranks_right = signed_rank_lower_loss_is_better(right_s).to_numpy(dtype=float)
    wmean = float(np.mean(w))
    if not np.isfinite(wmean) or wmean == 0.0:
        return dict(mas=np.nan, msdr=np.nan, expected_msdr=np.nan,
                    ranks_left=ranks_left, weights_array=None)
    wa = w / wmean
    e_msdr = expected_signed_msdr_weighted(wa, ranks_left)
    msdr = float(np.mean(wa * (ranks_left - ranks_right) ** 2))
    mas = float(1.0 - msdr / e_msdr)
    return dict(mas=mas, msdr=msdr, expected_msdr=e_msdr,
                ranks_left=ranks_left, weights_array=wa)


def mc_p(res, alpha, seed_parts):
    if res["weights_array"] is None or not np.isfinite(res["mas"]):
        return np.nan
    return float(monte_carlo_p_value(
        observed_mas=res["mas"], expected_msdr=res["expected_msdr"],
        ranks=res["ranks_left"], weights=res["weights_array"],
        null_type="signed", n_sims=N_SIMS, alpha=alpha,
        rng=np.random.default_rng(stable_seed(*seed_parts, round(alpha, 4))),
        chunk_size=CHUNK, add_one=True))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    per_seed_rows = []
    summary_rows = []
    concordance_rows = []

    for scenario in SCENARIOS:
        ins = load_insample(scenario)
        for model in MODELS:
            # store[(comp, loss)] = list over seeds of (left, right, weights)
            store: dict[tuple[str, str], list] = {}
            mas_by_key: dict[tuple[str, str], list] = {}
            for seed in SEEDS:
                sub = (ins[(ins["model"] == model) & (ins["seed"] == seed)]
                       .set_index("feature").reindex(FEATURES))
                if sub[["is_gpbsv_mse", "oos_gpbsv_mse"]].isna().any().any():
                    raise ValueError(
                        f"Missing IS/OOS rows for {scenario}/{model}/seed{seed} "
                        f"(features expected: {FEATURES}).")
                imp = load_importance(scenario, model, seed).reindex(FEATURES)
                for comp in COMPARISONS:
                    for loss in LOSSES:
                        left, right, w, lrt = get_vectors(sub, imp, comp, loss)
                        res = one_mas(left, right, w, lrt)
                        per_seed_rows.append(dict(
                            scenario=scenario, model=model, comparison=comp,
                            loss=loss, seed=seed, mas=res["mas"]))
                        store.setdefault((comp, loss), []).append((left, right, w))
                        mas_by_key.setdefault((comp, loss), []).append(res["mas"])

            for (comp, loss), lst in store.items():
                lefts = np.array([x[0] for x in lst])      # (25, 4)
                rights = np.array([x[1] for x in lst])
                wts = np.array([x[2] for x in lst])
                lrt = "signed" if comp == "a" else "ordinary"
                seed_mas = np.array(mas_by_key[(comp, loss)], dtype=float)

                # Single-number supplement: MAS on the seed-averaged feature vectors.
                res_avg = one_mas(lefts.mean(0), rights.mean(0), wts.mean(0), lrt)
                p050 = mc_p(res_avg, 0.5, (scenario, model, comp, loss))
                p067 = mc_p(res_avg, 2.0 / 3.0, (scenario, model, comp, loss))

                # Descriptive: Spearman over the 25x4 cells.
                spearman = (pd.Series(lefts.ravel())
                            .corr(pd.Series(rights.ravel()), method="spearman"))

                summary_rows.append(dict(
                    scenario=scenario, model=model, comparison=comp,
                    comparison_label=COMP_LABEL[comp], loss=loss,
                    n_seeds=len(seed_mas),
                    mean_mas=float(np.nanmean(seed_mas)),
                    sd_mas=float(np.nanstd(seed_mas, ddof=1)),
                    seedavg_mas=res_avg["mas"],
                    seedavg_p_alpha050=p050, seedavg_p_alpha0667=p067,
                    spearman_100cells=float(spearman),
                    normalizer_alpha=0.5))

            # Sign/quadrant concordance (GPBSV-vs-GPBSV; comparison-agnostic), per loss.
            for loss in LOSSES:
                lefts, rights, _ = zip(*store[("a", loss)])
                is_cells = np.array(lefts).ravel()    # IS GPBSV
                oos_cells = np.array(rights).ravel()  # OOS GPBSV
                concordance_rows.append(dict(
                    scenario=scenario, model=model, loss=loss, n_cells=is_cells.size,
                    trap_signature_share=float(np.mean((is_cells < 0) & (oos_cells > 0))),
                    sign_agreement_share=float(np.mean(np.sign(is_cells) == np.sign(oos_cells)))))

    per_seed = pd.DataFrame(per_seed_rows)
    summary = pd.DataFrame(summary_rows)
    concordance = pd.DataFrame(concordance_rows)

    per_seed.to_csv(OUT_DIR / "sim_mas_is_vs_oos_gpbsv_per_seed.csv", index=False)
    summary.to_csv(OUT_DIR / "sim_mas_is_vs_oos_gpbsv_summary.csv", index=False)
    concordance.to_csv(OUT_DIR / "sim_mas_is_vs_oos_gpbsv_concordance.csv", index=False)

    # Console headline: mean MAS (MSE) per comparison x scenario x model.
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 50)
    mse = summary[summary["loss"] == "mse"]
    print("\n=== Mean per-seed MAS (MSE), by comparison / scenario / model ===")
    for comp in COMPARISONS:
        blk = mse[mse["comparison"] == comp]
        tab = blk.pivot(index="scenario", columns="model", values="mean_mas")[MODELS]
        print(f"\n[{comp}] {COMP_LABEL[comp]}")
        print(tab.round(3).to_string())
    print("\n=== Trap signature (MSE): share of seed-feature cells with IS<0 & OOS>0 ===")
    ct = concordance[concordance["loss"] == "mse"].pivot(
        index="scenario", columns="model", values="trap_signature_share")[MODELS]
    print(ct.round(3).to_string())
    print(f"\nWrote 3 CSVs to: {OUT_DIR}")


if __name__ == "__main__":
    main()
