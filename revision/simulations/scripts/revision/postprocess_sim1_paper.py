"""
sim1_postprocess_paper_outputs.py

Post-process Simulation 1 outputs (NO re-running Monte Carlo) into:

1) paper_summary.csv + paper_summary_table.tex (plain-language metrics)
2) fig_importance_vs_gpbsv.pdf (2x3 panel bars: TS-Shapley-VI vs GPBSV)
3) fig_x2_scatter.pdf (optional, but very communicative)

Assumes you already have, per scenario folder:
  - grand_summary.csv
  - table_{model}_seed{seed}.csv   (saved by your simulation code)

Run:
  python sim1_postprocess_paper_outputs.py --output_root "_sim1_master_out"

Optional:
  python sim1_postprocess_paper_outputs.py --output_root "_sim1_master_out" --no_x2_scatter
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Matplotlib is optional; if unavailable, plots will be skipped gracefully.
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


DEFAULT_MODEL_NAMES = ["ols", "enet", "rf"]
DEFAULT_FEATURES = ["x1", "x2", "x3", "x4"]
SCRIPT_DIR = Path(__file__).resolve().parent
SIMULATION_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_OUTPUT_ROOT = SIMULATION_ROOT / "outputs" / "sim1_master_out"


# ---------------------------
# Helpers: discovery + loading
# ---------------------------
def find_scenario_dirs(output_root: Path) -> List[Path]:
    """Find scenario directories that contain a grand_summary.csv."""
    if not output_root.exists():
        raise FileNotFoundError(f"Output root not found: {output_root}")

    dirs = []
    for p in output_root.iterdir():
        if p.is_dir() and (p / "grand_summary.csv").exists():
            dirs.append(p)
    return sorted(dirs)


def load_grand_summary(scenario_dir: Path) -> pd.DataFrame:
    path = scenario_dir / "grand_summary.csv"
    df = pd.read_csv(path)
    # Defensive typing
    if "seed" in df.columns:
        df["seed"] = df["seed"].astype(int)
    if "model" in df.columns:
        df["model"] = df["model"].astype(str)
    return df


def parse_seed_from_filename(fname: str) -> Optional[int]:
    m = re.search(r"seed(\d+)", fname)
    return int(m.group(1)) if m else None


def build_feature_long_from_tables(
    scenario_dir: Path,
    model_names: List[str],
) -> pd.DataFrame:
    """
    Construct a long (tidy) feature-level dataset by reading the per-seed tables.

    Output columns include:
      scenario, seed, model, feature, true_role,
      oShapley_VI, VI_rank, GPBSV_MSE, GPBSV_RMSE, perm_import_train, pdp_range, ...
    """
    rows = []
    scenario_name = scenario_dir.name

    for model in model_names:
        # pattern: table_{model}_seed{seed}.csv
        for csv_path in scenario_dir.glob(f"table_{model}_seed*.csv"):
            seed = parse_seed_from_filename(csv_path.name)
            if seed is None:
                continue

            tbl = pd.read_csv(csv_path, index_col=0)
            tbl = tbl.copy()
            tbl["feature"] = tbl.index.astype(str)
            tbl = tbl.reset_index(drop=True)

            tbl.insert(0, "scenario", scenario_name)
            tbl.insert(1, "seed", seed)
            tbl.insert(2, "model", model)

            rows.append(tbl)

    if not rows:
        raise FileNotFoundError(
            f"No per-seed tables found in {scenario_dir}. "
            f"Expected files like table_ols_seed100.csv."
        )

    out = pd.concat(rows, axis=0, ignore_index=True)
    # Normalize column names used by your simulation outputs
    # (Already matches your code, but keep a light guard.)
    rename_map = {
        "oShapley_VI": "oShapley_VI",
        "GPBSV_MSE": "GPBSV_MSE",
        "GPBSV_RMSE": "GPBSV_RMSE",
        "perm_import_train": "perm_import_train",
        "pdp_range": "pdp_range",
        "VI_rank": "VI_rank",
        "true_role": "true_role",
    }
    for k, v in rename_map.items():
        if k in out.columns:
            out.rename(columns={k: v}, inplace=True)

    return out


# ---------------------------
# Metrics: paper-friendly events
# ---------------------------
def _rank_x2_cols_for_model(sub: pd.DataFrame) -> List[str]:
    """Rank columns present for this model (non-all-NaN)."""
    cols = [c for c in sub.columns if c.startswith("rank_x2_")]
    cols = [c for c in cols if sub[c].notna().any()]
    return cols


def compute_x2_top2_all_classical(sub: pd.DataFrame) -> pd.Series:
    """
    For each row (seed), returns True if x2 is top-2 in ALL classical proxy measures
    available for that model.
    """
    cols = _rank_x2_cols_for_model(sub)
    if not cols:
        return pd.Series(False, index=sub.index)

    def _row_ok(r: pd.Series) -> bool:
        vals = r[cols].dropna()
        if vals.empty:
            return False
        return bool((vals <= 2).all())

    return sub.apply(_row_ok, axis=1)


def compute_x2_rank1_all_classical(sub: pd.DataFrame) -> pd.Series:
    cols = _rank_x2_cols_for_model(sub)
    if not cols:
        return pd.Series(False, index=sub.index)

    def _row_ok(r: pd.Series) -> bool:
        vals = r[cols].dropna()
        if vals.empty:
            return False
        return bool((vals == 1).all())

    return sub.apply(_row_ok, axis=1)


def fmt_rate(k: int, n: int, *, latex: bool = False) -> str:
    pct = 100.0 * k / max(n, 1)
    if latex:
        return f"{k}/{n} ({pct:.0f}\\%)"
    return f"{k}/{n} ({pct:.0f}%)"


def build_paper_summary(
    grand: pd.DataFrame,
    feature_long: pd.DataFrame,
    *,
    eps: float,
    improve_pct_min: float,
    model_names: List[str],
    features: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      summary_num: numeric summary for analysis
      summary_disp: display strings for tables / LaTeX
    """
    scenario_name = str(grand["scenario"].iloc[0]) if "scenario" in grand.columns else "unknown"
    out_num = []
    out_disp = []

    for model in model_names:
        sub = grand[grand["model"] == model].copy()
        n = len(sub)
        if n == 0:
            continue

        # In-sample importance conditions
        x2_top2_oshapley = (sub["x2_oshapley_rank"] <= 2)
        x2_top2_all_classical = compute_x2_top2_all_classical(sub)
        important_insample = x2_top2_oshapley & x2_top2_all_classical

        # Out-of-sample harm conditions
        harmful_gpbsv = (sub["x2_pbsv_mse"] > eps)
        drop_improves = sub["drop_x2_improves"].astype(bool)
        harmful_oos = harmful_gpbsv & drop_improves

        # Paradox events
        paradox = important_insample & harmful_oos
        paradox_large_gain = paradox & (sub["improve_pct"] >= improve_pct_min)

        # Baseline check
        full_beats_baseline = sub["model_beats_baseline"].astype(bool)

        # Medians (robust to outliers)
        med_gpbsv = float(np.nanmedian(sub["x2_pbsv_mse"].values))
        med_improve = float(np.nanmedian(sub["improve_pct"].values))

        # Near-zero sanity checks for x3/x4 using feature_long
        fsub = feature_long[(feature_long["model"] == model)].copy()
        # Make sure we have GPBSV_MSE
        p_x3_near0 = np.nan
        p_x4_near0 = np.nan
        if "GPBSV_MSE" in fsub.columns:
            for feat in ["x3", "x4"]:
                tmp = fsub[fsub["feature"] == feat]
                if len(tmp) > 0:
                    p = float((tmp["GPBSV_MSE"].abs() <= eps).mean())
                    if feat == "x3":
                        p_x3_near0 = p
                    else:
                        p_x4_near0 = p

        # ENet alpha diagnostics
        enet_alpha_mean = np.nan
        enet_alpha_min = np.nan
        enet_alpha_max = np.nan
        if model == "enet" and "enet_alpha" in sub.columns:
            enet_alpha_mean = float(np.nanmean(sub["enet_alpha"]))
            enet_alpha_min = float(np.nanmin(sub["enet_alpha"]))
            enet_alpha_max = float(np.nanmax(sub["enet_alpha"]))

        # Counts
        k_imp = int(important_insample.sum())
        k_harm = int(harmful_oos.sum())
        k_par = int(paradox.sum())
        k_par10 = int(paradox_large_gain.sum())
        k_full = int(full_beats_baseline.sum())
        k_drop = int(drop_improves.sum())

        out_num.append(
            dict(
                scenario=scenario_name,
                model=model,
                n_seeds=n,
                p_important_insample=float(k_imp / n),
                p_harmful_oos=float(k_harm / n),
                p_paradox=float(k_par / n),
                p_paradox_large_gain=float(k_par10 / n),
                p_full_beats_baseline=float(k_full / n),
                p_drop_improves=float(k_drop / n),
                median_gpbsv_x2=med_gpbsv,
                median_improve_pct=med_improve,
                p_x3_near_zero=p_x3_near0,
                p_x4_near_zero=p_x4_near0,
                enet_alpha_mean=enet_alpha_mean,
                enet_alpha_min=enet_alpha_min,
                enet_alpha_max=enet_alpha_max,
            )
        )

        out_disp.append(
            dict(
                scenario=scenario_name,
                model=model.upper(),
                N=str(n),
                important_insample=fmt_rate(k_imp, n),
                harmful_oos=fmt_rate(k_harm, n),
                paradox=fmt_rate(k_par, n),
                paradox_ge10=fmt_rate(k_par10, n),
                full_beats_baseline=fmt_rate(k_full, n),
                drop_improves=fmt_rate(k_drop, n),
                median_gpbsv_x2=f"{med_gpbsv:+.3f}",
                median_improve_pct=f"{med_improve:+.1f}%",
                x3_near0=f"{100*p_x3_near0:.0f}%" if np.isfinite(p_x3_near0) else "NA",
                x4_near0=f"{100*p_x4_near0:.0f}%" if np.isfinite(p_x4_near0) else "NA",
                enet_alpha=f"{enet_alpha_mean:.4g} [{enet_alpha_min:.4g},{enet_alpha_max:.4g}]"
                if model == "enet" and np.isfinite(enet_alpha_mean)
                else "",
            )
        )

    summary_num = pd.DataFrame(out_num)
    summary_disp = pd.DataFrame(out_disp)
    return summary_num, summary_disp


# ---------------------------
# Figures
# ---------------------------
def make_feature_agg(feature_long: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """
    Aggregate to mean/median per (model, feature).
    """
    needed = ["model", "feature", "oShapley_VI", "GPBSV_MSE"]
    missing = [c for c in needed if c not in feature_long.columns]
    if missing:
        raise ValueError(f"feature_long missing columns: {missing}")

    agg = (
        feature_long.groupby(["model", "feature"])
        .agg(
            mean_oShapley=("oShapley_VI", "mean"),
            median_oShapley=("oShapley_VI", "median"),
            mean_GPBSV=("GPBSV_MSE", "mean"),
            median_GPBSV=("GPBSV_MSE", "median"),
        )
        .reset_index()
    )
    # Enforce feature order for convenience in plotting
    agg["feature"] = pd.Categorical(agg["feature"], categories=features, ordered=True)
    agg = agg.sort_values(["model", "feature"]).reset_index(drop=True)
    return agg


def plot_importance_vs_gpbsv(
    feature_agg: pd.DataFrame,
    scenario_name: str,
    outpath: Path,
    *,
    model_names: List[str],
    features: List[str],
    use_median: bool = False,
) -> None:
    if plt is None:
        print("Matplotlib unavailable: skipping plots.")
        return

    stat_imp = "median_oShapley" if use_median else "mean_oShapley"
    stat_gpb = "median_GPBSV" if use_median else "mean_GPBSV"

    fig, axes = plt.subplots(2, len(model_names), figsize=(4.2 * len(model_names), 6.0))

    for j, model in enumerate(model_names):
        sub = feature_agg[feature_agg["model"] == model].set_index("feature").reindex(features)

        # Row 1: TS-Shapley-VI
        ax1 = axes[0, j] if len(model_names) > 1 else axes[0]
        bars1 = ax1.bar(features, sub[stat_imp].values)
        ax1.set_title(f"{model.upper()}: TS-Shapley-VI")
        ax1.set_ylabel("Importance (avg over seeds)")
        ax1.set_xlabel("")
        ax1.tick_params(axis="x", rotation=0)

        # Highlight x2 with hatch (no color choices needed)
        for b, feat in zip(bars1, features):
            if feat == "x2":
                b.set_hatch("//")

        # Row 2: GPBSV
        ax2 = axes[1, j] if len(model_names) > 1 else axes[1]
        bars2 = ax2.bar(features, sub[stat_gpb].values)
        ax2.axhline(0.0, linewidth=1.0)
        ax2.set_title(f"{model.upper()}: GPBSV (MSE)")
        ax2.set_ylabel("Contribution to OOS loss\n(positive = harmful)")
        ax2.set_xlabel("")
        ax2.tick_params(axis="x", rotation=0)

        for b, feat in zip(bars2, features):
            if feat == "x2":
                b.set_hatch("//")

    fig.suptitle(f"{scenario_name}: In-sample importance vs out-of-sample loss attribution", y=1.02)
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def plot_x2_scatter(
    grand: pd.DataFrame,
    feature_long: pd.DataFrame,
    scenario_name: str,
    outpath: Path,
    *,
    model_names: List[str],
    improve_pct_min: float,
) -> None:
    if plt is None:
        print("Matplotlib unavailable: skipping plots.")
        return

    # Merge x2 info from feature_long with improve_pct from grand
    x2 = feature_long[feature_long["feature"] == "x2"][["seed", "model", "oShapley_VI", "GPBSV_MSE"]].copy()
    x2 = x2.merge(grand[["seed", "model", "improve_pct"]], on=["seed", "model"], how="left")

    fig, axes = plt.subplots(1, len(model_names), figsize=(4.2 * len(model_names), 3.6))

    if len(model_names) == 1:
        axes = [axes]

    for j, model in enumerate(model_names):
        ax = axes[j]
        sub = x2[x2["model"] == model].copy()
        if sub.empty:
            ax.set_title(f"{model.upper()}: no data")
            continue

        big = sub[sub["improve_pct"] >= improve_pct_min]
        small = sub[sub["improve_pct"] < improve_pct_min]

        # Same color (default), different markers to avoid reliance on color coding.
        ax.scatter(small["oShapley_VI"], small["GPBSV_MSE"], marker="o", alpha=0.8, label=f"Δ<{improve_pct_min:.0f}%")
        ax.scatter(big["oShapley_VI"], big["GPBSV_MSE"], marker="^", alpha=0.9, label=f"Δ≥{improve_pct_min:.0f}%")

        ax.axhline(0.0, linewidth=1.0)
        ax.set_title(model.upper())
        ax.set_xlabel("x2 TS-Shapley-VI (in-sample importance)")
        ax.set_ylabel("x2 GPBSV (MSE)\n(positive = harmful)")

        ax.legend(frameon=False, fontsize=9)

    fig.suptitle(f"{scenario_name}: x2 importance vs harm across Monte Carlo seeds", y=1.05)
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


# ---------------------------
# LaTeX table writer
# ---------------------------
def write_latex_table(summary_disp: pd.DataFrame, outpath: Path, caption: str, label: str) -> None:
    # Build a compact LaTeX table from the display df
    cols = [
        "model",
        "important_insample",
        "harmful_oos",
        "paradox",
        "paradox_ge10",
        "median_gpbsv_x2",
        "median_improve_pct",
        "full_beats_baseline",
    ]
    keep = [c for c in cols if c in summary_disp.columns]
    df = summary_disp[keep].copy()

    # Escape percent signs for LaTeX in all string cells
    for c in df.columns:
        df[c] = df[c].astype(str).str.replace("%", "\\%", regex=False)

    latex = df.to_latex(
        index=False,
        escape=False,
        column_format="l" + "c" * (len(df.columns) - 1),
        caption=caption,
        label=label,
        longtable=False,
        bold_rows=False,
    )

    outpath.write_text(latex, encoding="utf-8")


# ---------------------------
# Main driver
# ---------------------------
def postprocess_one_scenario(
    scenario_dir: Path,
    *,
    eps: float,
    improve_pct_min: float,
    model_names: List[str],
    features: List[str],
    make_x2_scatter: bool,
) -> None:
    scenario_name = scenario_dir.name
    print(f"\n--- Postprocessing: {scenario_name} ---")

    grand = load_grand_summary(scenario_dir)

    # Feature-long: use existing if present, else build from per-seed tables
    feature_long_path = scenario_dir / "feature_long.csv"
    if feature_long_path.exists():
        feature_long = pd.read_csv(feature_long_path)
    else:
        feature_long = build_feature_long_from_tables(scenario_dir, model_names=model_names)
        feature_long.to_csv(feature_long_path, index=False)

    # Feature aggregation (for plotting and sanity checks)
    feature_agg = make_feature_agg(feature_long, features=features)
    (scenario_dir / "feature_agg.csv").write_text(feature_agg.to_csv(index=False), encoding="utf-8")

    # Paper summary tables
    summary_num, summary_disp = build_paper_summary(
        grand,
        feature_long,
        eps=eps,
        improve_pct_min=improve_pct_min,
        model_names=model_names,
        features=features,
    )

    summary_num.to_csv(scenario_dir / "paper_summary.csv", index=False)
    summary_disp.to_csv(scenario_dir / "paper_summary_display.csv", index=False)

    write_latex_table(
        summary_disp,
        outpath=scenario_dir / "paper_summary_table.tex",
        caption=f"Monte Carlo summary ({scenario_name})",
        label=f"tab:{scenario_name}",
    )

    print("Wrote:")
    print(f"  - {scenario_dir / 'paper_summary.csv'}")
    print(f"  - {scenario_dir / 'paper_summary_table.tex'}")

    # Figures
    fig1 = scenario_dir / "fig_importance_vs_gpbsv.pdf"
    plot_importance_vs_gpbsv(
        feature_agg,
        scenario_name=scenario_name,
        outpath=fig1,
        model_names=model_names,
        features=features,
        use_median=False,
    )
    if plt is not None:
        print(f"  - {fig1}")

    if make_x2_scatter:
        fig2 = scenario_dir / "fig_x2_scatter.pdf"
        plot_x2_scatter(
            grand,
            feature_long,
            scenario_name=scenario_name,
            outpath=fig2,
            model_names=model_names,
            improve_pct_min=improve_pct_min,
        )
        if plt is not None:
            print(f"  - {fig2}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_root", type=str, default=str(DEFAULT_OUTPUT_ROOT))
    ap.add_argument("--eps", type=float, default=0.02)
    ap.add_argument("--improve_pct_min", type=float, default=10.0)
    ap.add_argument("--no_x2_scatter", action="store_true", help="Disable x2 scatter figure")
    args = ap.parse_args()

    output_root = Path(args.output_root).expanduser().resolve()
    scenario_dirs = find_scenario_dirs(output_root)

    if not scenario_dirs:
        raise FileNotFoundError(f"No scenario dirs with grand_summary.csv found under {output_root}")

    for sd in scenario_dirs:
        postprocess_one_scenario(
            sd,
            eps=float(args.eps),
            improve_pct_min=float(args.improve_pct_min),
            model_names=list(DEFAULT_MODEL_NAMES),
            features=list(DEFAULT_FEATURES),
            make_x2_scatter=not args.no_x2_scatter,
        )

    print("\nDone postprocessing.")


if __name__ == "__main__":
    main()
