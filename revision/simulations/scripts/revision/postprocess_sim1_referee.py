"""sim1_postprocess_referee.py

Post-processing for *Simulation 1*.

What it does
------------
Given the output folder created by ``simulation_1_final_master.py`` (the master
simulation script), this script creates "referee-ready" outputs:

1) Feature-level figures that compare GPBSV (performance attribution) to
   alternative importance proxies (e.g., OOB permutation VI, average partial
   effect, standardized coefficients, PDP range, and oShapley VI).

2) "Average" versions of the same figures (means across Monte Carlo seeds,
   with simple uncertainty bands).

3) Model-level MAS (model accordance score) summaries:
   - a compact CSV table (mean MAS, dispersion, and significance rates)
   - a MAS-vs-RMSE quadrant plot (mirrors the visualization in the Anatomy paper)
   - a MAS distribution plot

Expected folder structure
-------------------------
The master simulation writes:

  <OUTPUT_ROOT>/<scenario>/grand_summary.csv
  <OUTPUT_ROOT>/<scenario>/table_<model>_seed<seed>.csv

This script discovers all scenarios under OUTPUT_ROOT.

Run
---
  python sim1_postprocess_referee.py --output-root _sim1_master_out

Notes
-----
* This script **does not** re-run the simulation.
* It is safe to run multiple times; it overwrites figures/tables.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

FEATURES: List[str] = ["x1", "x2", "x3", "x4"]

# Per-feature visual encoding (no text labels on points).
FEATURE_MARKERS: Dict[str, str] = {
    "x1": "o",
    "x2": "s",
    "x3": "^",
    "x4": "D",
}

_cmap = plt.get_cmap("tab10")
FEATURE_COLORS: Dict[str, Tuple[float, float, float, float]] = {
    "x1": _cmap(0),
    "x2": _cmap(1),
    "x3": _cmap(2),
    "x4": _cmap(3),
}


# What to plot against GPBSV per model.
# The first element of each tuple is the column in the per-seed table.
# The second element is the human-readable label.
MODEL_METRICS: Dict[str, List[Tuple[str, str]]] = {
    "ols": [
        ("avg_abs_tstat", "|t|-stat"),
        ("avg_std_beta", "Std. coefficient"),
        ("oShapley_VI", "TS-Shapley-VI"),
        ("pdp_range", "PDP-VI"),
    ],
    "enet": [
        ("enet_std_coef", "Std. coefficient"),
        ("enet_ape_abs_coef", "|Coefficient|"),
        ("oShapley_VI", "TS-Shapley-VI"),
        ("pdp_range", "PDP-VI"),
    ],
    "rf": [
        ("rf_oob_vi", "OOB permutation VI"),
        ("rf_ape", "Average partial effect"),
        ("oShapley_VI", "TS-Shapley-VI"),
        ("pdp_range", "PDP-VI"),
    ],
}


# Scenario-specific plotting choices requested by you.
SKIP_OLS_FOR_SCENARIOS_IN_REFEREE_TOPFIG = {"break_b08_expanding"}


REFEREE_DIRNAME = "referee_outputs"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SIMULATION_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
DEFAULT_OUTPUT_ROOT = os.path.join(SIMULATION_ROOT, "outputs", "sim1_master_out")


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    m = np.nanmean(x)
    s = np.nanstd(x)
    if not np.isfinite(s) or s <= 1e-12:
        return x * np.nan
    return (x - m) / s


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _discover_scenario_dirs(output_root: str) -> List[str]:
    out: List[str] = []
    if not os.path.isdir(output_root):
        return out
    for name in sorted(os.listdir(output_root)):
        d = os.path.join(output_root, name)
        if not os.path.isdir(d):
            continue
        if os.path.exists(os.path.join(d, "grand_summary.csv")):
            out.append(d)
    return out


def _read_grand_summary(scenario_dir: str) -> pd.DataFrame:
    p = os.path.join(scenario_dir, "grand_summary.csv")
    df = pd.read_csv(p)
    # normalize types
    df["seed"] = df["seed"].astype(int)
    df["model"] = df["model"].astype(str)
    return df


def _read_seed_table(scenario_dir: str, model: str, seed: int) -> Optional[pd.DataFrame]:
    p = os.path.join(scenario_dir, f"table_{model}_seed{seed}.csv")
    if not os.path.exists(p):
        return None
    tbl = pd.read_csv(p, index_col=0)
    tbl.index.name = "feature"
    tbl = tbl.reset_index()
    tbl["seed"] = int(seed)
    tbl["model"] = str(model)
    return tbl


def _build_long_feature_df(scenario_dir: str, grand: pd.DataFrame) -> pd.DataFrame:
    """Long dataframe: one row per (seed, model, feature).

    It merges per-seed feature tables with the scenario/model summary metrics.
    """
    rows: List[pd.DataFrame] = []
    for seed, model in grand[["seed", "model"]].drop_duplicates().itertuples(index=False):
        tbl = _read_seed_table(scenario_dir, model=model, seed=int(seed))
        if tbl is None:
            continue

        # Keep a consistent feature order and only known features.
        tbl = tbl[tbl["feature"].isin(FEATURES)].copy()

        # Attach model-level columns we may want later.
        g = grand[(grand["seed"] == int(seed)) & (grand["model"] == str(model))].iloc[0]
        for col in [
            "mse_full",
            "mse_drop_x2",
            "improve_pct",
            "x2_pbsv_mse",
            "mas",
            "mas_p",
        ]:
            if col in g.index:
                tbl[col] = g[col]
        rows.append(tbl)

    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out["scenario"] = os.path.basename(scenario_dir)
    return out


def _normalize_importance_within_seed(df: pd.DataFrame, col: str) -> pd.Series:
    """Normalize a positive-valued importance proxy into a per-seed 'share'."""
    v = pd.to_numeric(df[col], errors="coerce").astype(float)
    denom = float(np.nansum(v.values))
    if not np.isfinite(denom) or denom <= 1e-18:
        return pd.Series(np.nan, index=df.index)
    return v / denom


def _to_long_metric_view(feature_df: pd.DataFrame) -> pd.DataFrame:
    """Convert wide per-feature tables into a long 'metric' view for plotting."""
    out_rows: List[pd.DataFrame] = []
    for model, metrics in MODEL_METRICS.items():
        dfm = feature_df[feature_df["model"] == model].copy()
        if dfm.empty:
            continue

        for col, label in metrics:
            if col not in dfm.columns:
                continue

            d = dfm[["scenario", "seed", "model", "feature", "GPBSV_MSE", col]].copy()
            d = d.rename(columns={col: "importance_raw"})
            d["importance_raw"] = pd.to_numeric(d["importance_raw"], errors="coerce").astype(float)
            # Normalize within each (scenario, seed, model) while preserving row alignment.
            group_sum = d.groupby(["scenario", "seed", "model"])["importance_raw"].transform(
                lambda s: float(np.nansum(s.values))
            )
            group_sum = group_sum.where(np.isfinite(group_sum) & (group_sum > 1e-18))
            d["importance"] = d["importance_raw"] / group_sum
            d["metric"] = label
            out_rows.append(d[["scenario", "seed", "model", "feature", "metric", "importance", "GPBSV_MSE"]])

    if not out_rows:
        return pd.DataFrame()
    out = pd.concat(out_rows, ignore_index=True)
    return out


def _feature_legend_handles() -> List[Line2D]:
    handles: List[Line2D] = []
    for f in FEATURES:
        handles.append(
            Line2D(
                [0],
                [0],
                marker=FEATURE_MARKERS[f],
                color="none",
                markerfacecolor=FEATURE_COLORS[f],
                markeredgecolor="none",
                markersize=7,
                linestyle="None",
                label=f,
            )
        )
    return handles


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------


def make_importance_vs_gpbsv_pdf(
    long_metrics: pd.DataFrame,
    *,
    scenario: str,
    out_pdf_path: str,
    skip_models: Optional[Sequence[str]] = None,
) -> None:
    """Multi-page PDF: one page per model, panels per metric."""
    skip_models = set(skip_models or [])

    with PdfPages(out_pdf_path) as pdf:
        for model in ["ols", "enet", "rf"]:
            if model in skip_models:
                continue
            dfm = long_metrics[(long_metrics["scenario"] == scenario) & (long_metrics["model"] == model)].copy()
            if dfm.empty:
                continue

            metrics = [m for m in MODEL_METRICS.get(model, [])]
            metric_labels = [metric_label for _, metric_label in metrics if metric_label in dfm["metric"].unique()]
            if not metric_labels:
                continue

            ncols = len(metric_labels)
            fig, axes = plt.subplots(1, ncols, figsize=(4.25 * ncols, 4.0), sharey=True)
            if ncols == 1:
                axes = [axes]

            for ax, metric in zip(axes, metric_labels):
                d = dfm[dfm["metric"] == metric]

                # Horizontal zero line: beneficial vs harmful.
                ax.axhline(0.0, linewidth=1.0, alpha=0.6)

                for f in FEATURES:
                    dd = d[d["feature"] == f]
                    if dd.empty:
                        continue
                    ax.scatter(
                        dd["importance"],
                        dd["GPBSV_MSE"],
                        s=22,
                        alpha=0.55,
                        marker=FEATURE_MARKERS[f],
                        c=[FEATURE_COLORS[f]],
                        edgecolors="none",
                    )

                ax.set_title(metric)
                ax.set_xlabel("Normalized importance")
                ax.set_xlim(-0.02, 1.02)
                ax.grid(True, alpha=0.2)

            axes[0].set_ylabel("GPBSV (ΔMSE)")
            fig.legend(
                handles=_feature_legend_handles(),
                loc="upper center",
                ncol=len(FEATURES),
                frameon=False,
                bbox_to_anchor=(0.5, 1.10),
            )
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


def make_importance_vs_gpbsv_avg_pdf(
    long_metrics: pd.DataFrame,
    *,
    scenario: str,
    out_pdf_path: str,
    skip_models: Optional[Sequence[str]] = None,
) -> None:
    """Average (across seeds) version of the importance-vs-GPBSV figure."""
    skip_models = set(skip_models or [])

    with PdfPages(out_pdf_path) as pdf:
        for model in ["ols", "enet", "rf"]:
            if model in skip_models:
                continue

            dfm = long_metrics[(long_metrics["scenario"] == scenario) & (long_metrics["model"] == model)].copy()
            if dfm.empty:
                continue

            # Aggregate across seeds.
            agg = (
                dfm.groupby(["metric", "feature"], as_index=False)
                .agg(
                    importance_mean=("importance", "mean"),
                    importance_std=("importance", "std"),
                    gpbsv_mean=("GPBSV_MSE", "mean"),
                    gpbsv_std=("GPBSV_MSE", "std"),
                    n=("seed", "nunique"),
                )
            )

            metric_labels = [
                metric_label
                for _, metric_label in MODEL_METRICS.get(model, [])
                if metric_label in agg["metric"].unique()
            ]
            if not metric_labels:
                continue

            ncols = len(metric_labels)
            fig, axes = plt.subplots(1, ncols, figsize=(4.25 * ncols, 4.0), sharey=True)
            if ncols == 1:
                axes = [axes]

            for ax, metric in zip(axes, metric_labels):
                a = agg[agg["metric"] == metric]
                ax.axhline(0.0, linewidth=1.0, alpha=0.6)

                for f in FEATURES:
                    af = a[a["feature"] == f]
                    if af.empty:
                        continue
                    x = float(af["importance_mean"].iloc[0])
                    y = float(af["gpbsv_mean"].iloc[0])
                    xerr = float(af["importance_std"].iloc[0]) if np.isfinite(af["importance_std"].iloc[0]) else 0.0
                    yerr = float(af["gpbsv_std"].iloc[0]) if np.isfinite(af["gpbsv_std"].iloc[0]) else 0.0

                    ax.errorbar(
                        [x],
                        [y],
                        xerr=[xerr],
                        yerr=[yerr],
                        fmt=FEATURE_MARKERS[f],
                        markersize=7,
                        capsize=3,
                        alpha=0.9,
                        color=FEATURE_COLORS[f],
                        markerfacecolor=FEATURE_COLORS[f],
                        markeredgecolor="none",
                    )

                ax.set_title(metric)
                ax.set_xlabel("Mean normalized importance")
                ax.set_xlim(-0.02, 1.02)
                ax.grid(True, alpha=0.2)

            axes[0].set_ylabel("Mean GPBSV (ΔMSE)")
            fig.legend(
                handles=_feature_legend_handles(),
                loc="upper center",
                ncol=len(FEATURES),
                frameon=False,
                bbox_to_anchor=(0.5, 1.10),
            )
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


def make_mas_quadrant_plot(
    grand: pd.DataFrame,
    *,
    scenario: str,
    out_path: str,
    skip_models: Optional[Sequence[str]] = None,
) -> None:
    """Quadrant plot: MAS (vertical) vs RMSE (horizontal, decreasing left-to-right)."""
    skip_models = set(skip_models or [])
    d = grand[grand["scenario"] == scenario].copy() if "scenario" in grand.columns else grand.copy()
    d = d[~d["model"].isin(skip_models)].copy()
    if d.empty:
        return

    d["rmse"] = np.sqrt(pd.to_numeric(d["mse_full"], errors="coerce").astype(float))
    d["mas"] = pd.to_numeric(d["mas"], errors="coerce").astype(float)

    # Z-scores across all seed-model rows in this scenario.
    d["rmse_z"] = _zscore(d["rmse"].values)
    d["mas_z"] = _zscore(d["mas"].values)

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    ax.axhline(0.0, linewidth=1.0, alpha=0.6)
    ax.axvline(0.0, linewidth=1.0, alpha=0.6)

    # Small points: each seed.
    model_order = [m for m in ["ols", "enet", "rf"] if m in d["model"].unique() and m not in skip_models]
    model_colors = {m: plt.get_cmap("tab10")(i + 4) for i, m in enumerate(model_order)}

    for m in model_order:
        dm = d[d["model"] == m]
        ax.scatter(
            dm["rmse_z"],
            dm["mas_z"],
            s=18,
            alpha=0.25,
            c=[model_colors[m]],
            edgecolors="none",
            label=m.upper(),
        )

    # Large points: model means.
    means = d.groupby("model", as_index=False).agg(rmse_z=("rmse_z", "mean"), mas_z=("mas_z", "mean"))
    for _, r in means.iterrows():
        m = str(r["model"])
        ax.scatter(
            [float(r["rmse_z"])],
            [float(r["mas_z"])],
            s=110,
            alpha=0.95,
            c=[model_colors.get(m, "black")],
            edgecolors="none",
        )
        ax.text(float(r["rmse_z"]) + 0.03, float(r["mas_z"]) + 0.03, m.upper(), fontsize=10)

    ax.set_xlabel("RMSE (Z-score) — decreasing →")
    ax.set_ylabel("MAS (Z-score)")
    ax.set_title(f"{scenario}: MAS vs RMSE (quadrant plot)")

    # IMPORTANT: decreasing RMSE from left to right (as in the paper).
    ax.invert_xaxis()
    ax.grid(True, alpha=0.2)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def make_mas_distribution_plot(
    grand: pd.DataFrame,
    *,
    scenario: str,
    out_path: str,
    skip_models: Optional[Sequence[str]] = None,
) -> None:
    """Boxplot + jittered points for MAS by model."""
    skip_models = set(skip_models or [])
    d = grand[grand["scenario"] == scenario].copy() if "scenario" in grand.columns else grand.copy()
    d = d[~d["model"].isin(skip_models)].copy()
    if d.empty:
        return

    d["mas"] = pd.to_numeric(d["mas"], errors="coerce").astype(float)
    model_order = [m for m in ["ols", "enet", "rf"] if m in d["model"].unique() and m not in skip_models]

    data = [d.loc[d["model"] == m, "mas"].dropna().values for m in model_order]
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ax.boxplot(data, labels=[m.upper() for m in model_order], showfliers=False)

    # Jitter points.
    rng = np.random.default_rng(123)
    for i, m in enumerate(model_order, start=1):
        y = d.loc[d["model"] == m, "mas"].dropna().values
        if y.size == 0:
            continue
        x = i + rng.normal(0.0, 0.06, size=y.size)
        ax.scatter(x, y, s=18, alpha=0.35, edgecolors="none")

    ax.axhline(0.0, linewidth=1.0, alpha=0.6)
    ax.set_ylabel("MAS")
    ax.set_title(f"{scenario}: distribution of MAS across seeds")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Tables
# -----------------------------------------------------------------------------


def make_referee_model_summary_table(grand: pd.DataFrame) -> pd.DataFrame:
    """One row per (scenario, model) with performance + MAS summary."""
    d = grand.copy()
    if "scenario" not in d.columns:
        # Some grand_summary.csv files already include scenario column; your master script does.
        d["scenario"] = "(unknown)"

    d["rmse_full"] = np.sqrt(pd.to_numeric(d["mse_full"], errors="coerce").astype(float))
    d["mas"] = pd.to_numeric(d["mas"], errors="coerce").astype(float)
    d["mas_p"] = pd.to_numeric(d.get("mas_p", np.nan), errors="coerce").astype(float)

    def frac_sig(p: pd.Series, level: float) -> float:
        p = pd.to_numeric(p, errors="coerce").astype(float)
        return float(np.mean(p < level))

    out = (
        d.groupby(["scenario", "model"], as_index=False)
        .agg(
            n_seeds=("seed", "nunique"),
            mse_full_mean=("mse_full", "mean"),
            mse_drop_mean=("mse_drop_x2", "mean"),
            improve_pct_mean=("improve_pct", "mean"),
            rmse_full_mean=("rmse_full", "mean"),
            mas_mean=("mas", "mean"),
            mas_std=("mas", "std"),
            mas_median=("mas", "median"),
            mas_sig_10=("mas_p", lambda s: frac_sig(s, 0.10)),
            mas_sig_05=("mas_p", lambda s: frac_sig(s, 0.05)),
            mas_sig_01=("mas_p", lambda s: frac_sig(s, 0.01)),
        )
        .sort_values(["scenario", "model"])
    )
    return out


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--output-root",
        type=str,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root folder created by the master simulation script.",
    )
    args = ap.parse_args()

    output_root = os.path.abspath(args.output_root)
    scenario_dirs = _discover_scenario_dirs(output_root)
    if not scenario_dirs:
        raise SystemExit(f"No scenario folders with grand_summary.csv found under: {output_root}")

    # Build a single 'grand' across all scenarios (useful for a global summary table).
    grands: List[pd.DataFrame] = []
    feature_tables: List[pd.DataFrame] = []

    for scen_dir in scenario_dirs:
        scen = os.path.basename(scen_dir)
        grand = _read_grand_summary(scen_dir)

        # Ensure scenario column exists (your master script includes it; keep robust).
        if "scenario" not in grand.columns:
            grand["scenario"] = scen

        grands.append(grand)
        feature_tables.append(_build_long_feature_df(scen_dir, grand))

    grand_all = pd.concat(grands, ignore_index=True)
    feature_all = pd.concat([d for d in feature_tables if not d.empty], ignore_index=True) if feature_tables else pd.DataFrame()

    # Long metric view used for plotting.
    long_metrics = _to_long_metric_view(feature_all) if not feature_all.empty else pd.DataFrame()

    # Global summary table (all scenarios).
    global_referee_dir = _ensure_dir(os.path.join(output_root, REFEREE_DIRNAME))
    model_summary = make_referee_model_summary_table(grand_all)
    model_summary.to_csv(os.path.join(global_referee_dir, "referee_model_summary.csv"), index=False)

    # Scenario-specific outputs.
    for scen_dir in scenario_dirs:
        scen = os.path.basename(scen_dir)
        ref_dir = _ensure_dir(os.path.join(scen_dir, REFEREE_DIRNAME))

        # Save scenario-specific model summary table.
        scen_summary = model_summary[model_summary["scenario"] == scen].copy()
        scen_summary.to_csv(os.path.join(ref_dir, "referee_model_summary.csv"), index=False)

        # Skip-model logic requested by you.
        skip_models = []
        if scen in SKIP_OLS_FOR_SCENARIOS_IN_REFEREE_TOPFIG:
            skip_models = ["ols"]

        # Importance-vs-GPBSV figures (per-feature).
        if not long_metrics.empty:
            out_pdf = os.path.join(ref_dir, "fig_importance_vs_gpbsv.pdf")
            make_importance_vs_gpbsv_pdf(
                long_metrics,
                scenario=scen,
                out_pdf_path=out_pdf,
                skip_models=skip_models,
            )

            out_pdf_avg = os.path.join(ref_dir, "fig_importance_vs_gpbsv_avg.pdf")
            make_importance_vs_gpbsv_avg_pdf(
                long_metrics,
                scenario=scen,
                out_pdf_path=out_pdf_avg,
                skip_models=skip_models,
            )

        # MAS figures (model-level).
        grand_scen = grand_all[grand_all["scenario"] == scen].copy()
        make_mas_quadrant_plot(
            grand_scen,
            scenario=scen,
            out_path=os.path.join(ref_dir, "fig_mas_quadrant.pdf"),
            # NOTE: we don't automatically skip OLS here; change if you want.
            skip_models=None,
        )
        make_mas_distribution_plot(
            grand_scen,
            scenario=scen,
            out_path=os.path.join(ref_dir, "fig_mas_distribution.pdf"),
            skip_models=None,
        )

    print("Referee post-processing complete.")
    print(f"Global referee outputs: {global_referee_dir}")
    print("Per-scenario referee outputs are in each scenario folder:")
    for scen_dir in scenario_dirs:
        print("  -", os.path.join(scen_dir, REFEREE_DIRNAME))


if __name__ == "__main__":
    main()
