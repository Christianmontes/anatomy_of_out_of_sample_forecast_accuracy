"""
===============================================================================
PBSV Convergence Analysis with Uncertainty Bands (Monte Carlo error ribbons)
===============================================================================

What this script adds vs the original convergence script
--------------------------------------------------------
Your original convergence plots show PBSV estimates as a function of M (number of
Shapley Monte Carlo iterations). However, they do not show *how uncertain* the
estimate is at each M.

This script adds uncertainty bands ("ribbons") around the PBSV curves by:
  - fixing the data and forecasting models,
  - repeating the Shapley/PBSV estimation R times for each M with different RNG
    seeds (only the Shapley sampling changes),
  - summarizing across replicates by mean + central quantile band (default: 16-84%).

Interpretation
--------------
For each predictor p and model:
  - The line is the mean PBSV estimate across replicates.
  - The shaded area is the dispersion across replicates (Monte Carlo error).
  - The band should shrink as M increases (approx O(1/sqrt(M))).

Crowding / overlap
------------------
Overlaying 3 models + 3 shaded bands per panel can be crowded when curves are close.
To handle this, we save:
  (A) an overlay figure (all models in each panel), and
  (B) separate per-model figures (cleaner ribbons, recommended for appendix).

Dependencies
------------
  anatomy, numpy, pandas, matplotlib, scikit-learn, xgboost

Run
---
  python pbsv_convergence_with_bands.py
"""

import argparse
import contextlib
import io
import os
import random
import warnings
from dataclasses import dataclass, replace
from typing import Dict, List, Tuple

from tqdm import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from anatomy import (
    Anatomy,
    AnatomyModel,
    AnatomyModelCombination,
    AnatomyModelOutputTransformer,
    AnatomyModelProvider,
    AnatomySubsets,
)

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import xgboost as xgb

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SIMULATION_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
DEFAULT_OUT_DIR = os.path.join(SIMULATION_ROOT, "outputs", "convergence_bands")


# =============================================================================
# Plot styling (keep readable in papers)
# =============================================================================
plt.style.use("default")
plt.rcParams["figure.dpi"] = 110
plt.rcParams["font.size"] = 9
plt.rcParams["axes.labelsize"] = 10
plt.rcParams["axes.titlesize"] = 11
plt.rcParams["legend.fontsize"] = 8
plt.rcParams["xtick.labelsize"] = 8
plt.rcParams["ytick.labelsize"] = 8
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.30
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["figure.facecolor"] = "white"


# =============================================================================
# Configuration
# =============================================================================
@dataclass(frozen=True)
class Config:
    out_dir: str = DEFAULT_OUT_DIR

    base_seed: int = 1338

    # Replicates for the Monte Carlo uncertainty band
    n_replicates: int = 8

    n_jobs: int = -1  # -1 = use all available cores

    # DGP + sample sizes
    n_samples: int = 500
    n_predictors: int = 6
    initial_window: int = 200

    # M grid
    M_values: Tuple[int, ...] = (3, 6, 8, 12, 18, 30, 40, 60, 80, 100, 200, 500, 1000)

    # Models
    models: Tuple[str, ...] = ("enet", "rf", "xgboost")

    # Bands: central quantiles
    q_low: float = 0.16
    q_high: float = 0.84

    # Experiments to run
    run_all_experiments: bool = True  # set False if you only want Friedman rolling


CFG = Config()


def parse_args(base_cfg: Config) -> Config:
    parser = argparse.ArgumentParser(description="Run the GPBSV convergence study with uncertainty bands.")
    parser.add_argument("--out-dir", type=str, default=base_cfg.out_dir)
    parser.add_argument("--main-only", action="store_true", help="Run only the main Friedman/rolling experiment.")
    parser.add_argument("--smoke", action="store_true", help="Use a smaller, faster configuration for verification.")
    args = parser.parse_args()

    cfg = replace(
        base_cfg,
        out_dir=os.path.abspath(args.out_dir),
        run_all_experiments=not args.main_only,
    )

    if args.smoke:
        cfg = replace(
            cfg,
            n_replicates=2,
            n_samples=90,
            initial_window=40,
            M_values=(3, 6),
            run_all_experiments=False,
        )

    return cfg


# =============================================================================
# Utilities
# =============================================================================
def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))


def generate_ar1_predictors(
    n_samples: int,
    n_predictors: int,
    ar_coefs: np.ndarray,
    *,
    seed: int,
    noise_std: float = 1.0,
) -> np.ndarray:
    """
    Generate predictors as AR(1) processes with unit unconditional variance.

    X[t,j] = rho_j * X[t-1,j] + sqrt(1-rho_j^2) * eps, eps ~ N(0, noise_std^2)

    If noise_std=1.0, unconditional variance is 1 (stationary).
    """
    rng = np.random.default_rng(int(seed))
    X = np.zeros((n_samples, n_predictors), dtype=float)
    X[0, :] = rng.normal(0.0, noise_std, size=n_predictors)

    for t in range(1, n_samples):
        eps = rng.normal(0.0, noise_std, size=n_predictors)
        X[t, :] = ar_coefs * X[t - 1, :] + np.sqrt(1.0 - ar_coefs**2) * eps

    return X


def dgp_friedman1(*, seed: int, n_samples: int = 500, n_predictors: int = 6, noise_std: float = 0.1) -> pd.DataFrame:
    """
    Modified Friedman #1 in time-series setting.
    Predictors are AR(1) with decreasing persistence.
    Then min-max scaled to [0,1] for the Friedman function.
    """
    ar_coefs = np.array([0.8, 0.7, 0.6, 0.5, 0.4, 0.2], dtype=float)
    X = generate_ar1_predictors(n_samples, n_predictors, ar_coefs, seed=seed)

    # min-max scale per column to [0,1]
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X = (X - X_min) / (X_max - X_min + 1e-12)

    rng = np.random.default_rng(int(seed + 999))
    y = (
        15.0 * np.sin(np.pi * X[:, 0] * X[:, 1])
        + 10.0 * (X[:, 2] - 0.5) ** 2
        + 6.0 * X[:, 3]
        + 3.0 * X[:, 4]
        + 1.0 * X[:, 5]
        + rng.normal(0.0, noise_std, size=n_samples)
    )

    df = pd.DataFrame(X, columns=[f"x{i+1}" for i in range(n_predictors)])
    df["y"] = y
    df.index = pd.date_range("2021-01-01", periods=n_samples).map(lambda d: d.date())
    return df


def dgp_polynomial(*, seed: int, n_samples: int = 500, n_predictors: int = 6, noise_std: float = 0.5) -> pd.DataFrame:
    """
    Polynomial/interactions/periodic component.
    Predictors are AR(1) then standardized to mean 0 / var 1.
    """
    ar_coefs = np.array([0.8, 0.7, 0.6, 0.5, 0.4, 0.2], dtype=float)
    X = generate_ar1_predictors(n_samples, n_predictors, ar_coefs, seed=seed)

    # standardize each column
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)

    rng = np.random.default_rng(int(seed + 999))
    y = (
        2.0 * X[:, 0] ** 2
        + 1.5 * X[:, 1] ** 2
        + 1.2 * X[:, 0] * X[:, 1]
        + 0.8 * X[:, 2] ** 3
        + 0.5 * X[:, 3] * X[:, 4]
        + 0.2 * np.sin(2.0 * np.pi * X[:, 5])
        + rng.normal(0.0, noise_std, size=n_samples)
    )

    df = pd.DataFrame(X, columns=[f"x{i+1}" for i in range(n_predictors)])
    df["y"] = y
    df.index = pd.date_range("2021-01-01", periods=n_samples).map(lambda d: d.date())
    return df


def dgp_threshold(*, seed: int, n_samples: int = 500, n_predictors: int = 6, noise_std: float = 0.3) -> pd.DataFrame:
    """
    Threshold + periodic components, decreasing importance.
    Predictors are AR(1) then min-max scaled to [-2,2].
    """
    ar_coefs = np.array([0.8, 0.7, 0.6, 0.5, 0.4, 0.2], dtype=float)
    X = generate_ar1_predictors(n_samples, n_predictors, ar_coefs, seed=seed)

    # scale to [-2,2]
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X = 4.0 * (X - X_min) / (X_max - X_min + 1e-12) - 2.0

    rng = np.random.default_rng(int(seed + 999))
    y = (
        8.0 * (X[:, 0] > 0.0) * X[:, 1]
        + 5.0 * np.sin(4.0 * np.pi * X[:, 2]) * (X[:, 0] > 0.5)
        + 3.0 * X[:, 3] * np.cos(2.0 * np.pi * X[:, 2])
        + 1.5 * np.exp(-X[:, 4] ** 2)
        + 0.5 * X[:, 5]
        + rng.normal(0.0, noise_std, size=n_samples)
    )

    df = pd.DataFrame(X, columns=[f"x{i+1}" for i in range(n_predictors)])
    df["y"] = y
    df.index = pd.date_range("2021-01-01", periods=n_samples).map(lambda d: d.date())
    return df


def generate_data(dgp_type: str, window_type: str, *, cfg: Config) -> Tuple[pd.DataFrame, AnatomySubsets]:
    if dgp_type == "friedman1":
        xy = dgp_friedman1(seed=cfg.base_seed, n_samples=cfg.n_samples, n_predictors=cfg.n_predictors)
    elif dgp_type == "polynomial":
        xy = dgp_polynomial(seed=cfg.base_seed, n_samples=cfg.n_samples, n_predictors=cfg.n_predictors)
    elif dgp_type == "threshold":
        xy = dgp_threshold(seed=cfg.base_seed, n_samples=cfg.n_samples, n_predictors=cfg.n_predictors)
    else:
        raise ValueError(f"Unknown DGP type: {dgp_type}")

    estimation_type = (
        AnatomySubsets.EstimationType.ROLLING
        if window_type == "rolling"
        else AnatomySubsets.EstimationType.EXPANDING
    )

    subsets = AnatomySubsets.generate(
        index=xy.index,
        initial_window=int(cfg.initial_window),
        estimation_type=estimation_type,
        periods=1,
        gap=0,
    )
    return xy, subsets


# =============================================================================
# Model trainers (return AnatomyModel wrappers)
# =============================================================================
def train_enet(x_train: pd.DataFrame, y_train: pd.Series, *, seed: int) -> AnatomyModel:
    """
    Elastic Net with alpha chosen by TimeSeriesSplit CV, using a Pipeline
    so scaling is fold-safe.
    """
    alphas = [0.0001, 0.001, 0.01, 0.1]
    l1_ratio = 0.5
    cv = TimeSeriesSplit(n_splits=3)

    best_alpha = float(alphas[0])
    best_score = -np.inf

    for a in alphas:
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("enet", ElasticNet(alpha=float(a), l1_ratio=float(l1_ratio), max_iter=5000, random_state=int(seed))),
            ]
        )
        score = cross_val_score(pipe, x_train, y_train, cv=cv, scoring="neg_mean_squared_error").mean()
        if score > best_score:
            best_score = float(score)
            best_alpha = float(a)

    final_pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("enet", ElasticNet(alpha=float(best_alpha), l1_ratio=float(l1_ratio), max_iter=5000, random_state=int(seed))),
        ]
    ).fit(x_train, y_train)

    feat_cols = list(x_train.columns)

    def pred_fn(xs: np.ndarray) -> np.ndarray:
        xs_df = pd.DataFrame(xs, columns=feat_cols)
        return np.asarray(final_pipe.predict(xs_df)).ravel()

    return AnatomyModel(pred_fn)


def train_rf(x_train: pd.DataFrame, y_train: pd.Series, *, seed: int) -> AnatomyModel:
    """
    Random Forest (fixed hyperparameters).
    Scaling is not required for trees, so we skip it for speed.
    """
    feat_cols = list(x_train.columns)

    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=5,
        min_samples_split=10,
        random_state=int(seed),
    ).fit(x_train, y_train)

    def pred_fn(xs: np.ndarray) -> np.ndarray:
        xs_df = pd.DataFrame(xs, columns=feat_cols)
        return np.asarray(rf.predict(xs_df)).ravel()

    return AnatomyModel(pred_fn)


def train_xgboost(x_train: pd.DataFrame, y_train: pd.Series, *, seed: int) -> AnatomyModel:
    """
    XGBoost regressor (fixed hyperparameters).
    """
    feat_cols = list(x_train.columns)

    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        objective="reg:squarederror",
        random_state=int(seed),
    ).fit(x_train, y_train)

    def pred_fn(xs: np.ndarray) -> np.ndarray:
        xs_df = pd.DataFrame(xs, columns=feat_cols)
        return np.asarray(model.predict(xs_df)).ravel()

    return AnatomyModel(pred_fn)


# =============================================================================
# Provider builder with caching (critical for speed with replicates)
# =============================================================================
def build_cached_provider(
    xy: pd.DataFrame,
    subsets: AnatomySubsets,
    models_to_use: List[str],
    *,
    model_seed: int,
) -> AnatomyModelProvider:
    """
    Build an AnatomyModelProvider whose mapper caches per-(model,period) fitted models.
    This avoids re-training forecasting models for every replicate and M.
    """
    feat_cols = [c for c in xy.columns if c != "y"]

    # caches
    train_test_cache: Dict[int, Tuple[pd.DataFrame, pd.DataFrame]] = {}
    model_cache: Dict[Tuple[str, int], AnatomyModel] = {}

    def mapper(key: AnatomyModelProvider.PeriodKey) -> AnatomyModelProvider.PeriodValue:
        period = int(key.period)
        model_name = str(key.model_name)

        if period not in train_test_cache:
            train = xy.iloc[subsets.get_train_subset(period)]
            test = xy.iloc[subsets.get_test_subset(period)]
            train_test_cache[period] = (train, test)

        train, test = train_test_cache[period]

        cache_key = (model_name, period)
        if cache_key not in model_cache:
            x_train = train[feat_cols]
            y_train = train["y"]

            # IMPORTANT: seed depends on model+period so models are deterministic given data,
            # and does NOT change across Shapley replicates.
            this_seed = int(model_seed + 10_000 * (models_to_use.index(model_name) + 1) + period)

            if model_name == "enet":
                model_cache[cache_key] = train_enet(x_train, y_train, seed=this_seed)
            elif model_name == "rf":
                model_cache[cache_key] = train_rf(x_train, y_train, seed=this_seed)
            elif model_name == "xgboost":
                model_cache[cache_key] = train_xgboost(x_train, y_train, seed=this_seed)
            else:
                raise ValueError(f"Unknown model: {model_name}")

        return AnatomyModelProvider.PeriodValue(train, test, model_cache[cache_key])

    provider = AnatomyModelProvider(
        n_periods=subsets.n_periods,
        n_features=len(feat_cols),
        model_names=list(models_to_use),
        y_name="y",
        provider_fn=mapper,
    )
    return provider


# =============================================================================
# PBSV computation (RMSE)
# =============================================================================
def compute_pbsv_rmse(anatomy_obj: Anatomy, models_to_use: List[str]) -> pd.DataFrame:
    groups = {m: [m] for m in models_to_use}

    def transform_rmse(y_hat, y):
        return float(np.sqrt(np.mean((y - y_hat) ** 2)))

    rmse_df = anatomy_obj.explain(
        model_sets=AnatomyModelCombination(groups=groups),
        transformer=AnatomyModelOutputTransformer(transform=transform_rmse),
    )
    return rmse_df


# =============================================================================
# Main convergence routine with replicate bands
# =============================================================================
def run_convergence_with_bands(dgp_type: str, window_type: str, *, cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      - raw_long: one row per (model, predictor, M, replicate) with pbsv value
      - summary: aggregated mean and quantiles per (model, predictor, M)
    """
    print(f"\n>>> PBSV Convergence w/ bands: DGP={dgp_type} | window={window_type}")
    print("-" * 78)

    xy, subsets = generate_data(dgp_type, window_type, cfg=cfg)
    models = list(cfg.models)

    # Build cached provider once (models retrained over periods once)
    provider = build_cached_provider(xy, subsets, models, model_seed=cfg.base_seed)

    feat_cols = [c for c in xy.columns if c != "y"]

    rows = []

    total_iters = len(cfg.M_values) * cfg.n_replicates
    with tqdm(total=total_iters, desc=f"{dgp_type}/{window_type}", unit="iter", leave=True) as pbar:
        # Loop over M and replicates: only Shapley sampling changes
        for M in cfg.M_values:
            for r in range(cfg.n_replicates):
                pbar.set_postfix(M=M, rep=f"{r+1}/{cfg.n_replicates}")

                shapley_seed = int(cfg.base_seed + 1_000_000 + 10_000 * r + 17 * M)
                seed_everything(shapley_seed)

                save_path = os.path.join(cfg.out_dir, f"tmp_anatomy_{dgp_type}_{window_type}_M{M}_rep{r}.bin")
                try:
                    anatomy_obj = Anatomy(provider=provider, n_iterations=int(M))
                    with contextlib.redirect_stdout(io.StringIO()):
                        anatomy_obj.precompute(n_jobs=int(cfg.n_jobs), save_path=save_path)
                        anatomy_obj = Anatomy.load(save_path)
                        pbsv_df = compute_pbsv_rmse(anatomy_obj, models)

                    # pbsv_df.loc[model] is typically a 1-row df; we extract row 0 and drop base_contribution
                    for model in models:
                        block = pbsv_df.loc[model]
                        if isinstance(block, pd.DataFrame):
                            vals = block.iloc[0].drop("base_contribution", errors="ignore")
                        else:
                            vals = block.drop("base_contribution", errors="ignore")

                        for p in feat_cols:
                            rows.append(
                                dict(
                                    dgp=dgp_type,
                                    window=window_type,
                                    model=model,
                                    predictor=p,
                                    M=int(M),
                                    replicate=int(r),
                                    pbsv=float(vals[p]),
                                )
                            )
                finally:
                    # cleanup tmp file
                    try:
                        if os.path.exists(save_path):
                            os.remove(save_path)
                    except OSError:
                        pass

                pbar.update(1)

    raw_long = pd.DataFrame(rows)

    # Aggregate
    def qfun(q):
        return lambda x: float(np.quantile(x, q))

    summary = (
        raw_long.groupby(["dgp", "window", "model", "predictor", "M"])["pbsv"]
        .agg(
            mean="mean",
            std="std",
            q_low=qfun(cfg.q_low),
            q_high=qfun(cfg.q_high),
        )
        .reset_index()
    )
    return raw_long, summary


# =============================================================================
# Plotting
# =============================================================================
MODEL_STYLES = {
    "enet": dict(color="blue", marker="o", linestyle="-", label="ENet"),
    "rf": dict(color="green", marker="s", linestyle="--", label="RF"),
    "xgboost": dict(color="red", marker="^", linestyle="-.", label="XGBoost"),
}


def plot_overlay_with_bands(summary: pd.DataFrame, dgp_type: str, window_type: str, *, cfg: Config) -> None:
    """
    2x3 panels (predictors). Each panel overlays models with mean line + band.
    Can look crowded if curves overlap; we therefore also output per-model figures.
    """
    predictors = [f"x{i+1}" for i in range(cfg.n_predictors)]
    ordinal = ["First", "Second", "Third", "Fourth", "Fifth", "Sixth"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for j, pred in enumerate(predictors):
        ax = axes[j]
        subp = summary[(summary["dgp"] == dgp_type) & (summary["window"] == window_type) & (summary["predictor"] == pred)]

        for model in cfg.models:
            sm = subp[subp["model"] == model].sort_values("M")
            if sm.empty:
                continue

            style = MODEL_STYLES[model]
            x = sm["M"].values
            y = sm["mean"].values
            lo = sm["q_low"].values
            hi = sm["q_high"].values

            ax.plot(
                x, y,
                marker=style["marker"],
                linestyle=style["linestyle"],
                color=style["color"],
                linewidth=2,
                markersize=4,
                label=style["label"],
            )
            # light band to reduce overplotting
            ax.fill_between(x, lo, hi, color=style["color"], alpha=0.10, linewidth=0)

        ax.set_title(f"{ordinal[j]} Predictor", fontweight="bold")
        ax.set_xlabel("M")
        ax.set_ylabel("PBSV (RMSE)")
        ax.set_xscale("log")
        ax.axhline(0.0, color="gray", linewidth=0.8, alpha=0.5)
        ax.grid(True, which="both", alpha=0.3)

        if j == 0:
            ax.legend(loc="best")

    plt.tight_layout()

    base = os.path.join(cfg.out_dir, f"convergence_bands_overlay_{dgp_type}_{window_type}")
    plt.savefig(base + ".png", dpi=300, bbox_inches="tight")
    plt.savefig(base + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {base}.png/.pdf")


def plot_per_model_with_bands(summary: pd.DataFrame, dgp_type: str, window_type: str, *, cfg: Config) -> None:
    """
    Separate 2x3 figure per model: cleaner visualization of the shrinking band.
    """
    predictors = [f"x{i+1}" for i in range(cfg.n_predictors)]
    ordinal = ["First", "Second", "Third", "Fourth", "Fifth", "Sixth"]

    for model in cfg.models:
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes = axes.flatten()

        style = MODEL_STYLES[model]

        for j, pred in enumerate(predictors):
            ax = axes[j]
            sm = summary[
                (summary["dgp"] == dgp_type)
                & (summary["window"] == window_type)
                & (summary["model"] == model)
                & (summary["predictor"] == pred)
            ].sort_values("M")

            x = sm["M"].values
            y = sm["mean"].values
            lo = sm["q_low"].values
            hi = sm["q_high"].values

            ax.plot(
                x, y,
                marker=style["marker"],
                linestyle=style["linestyle"],
                color=style["color"],
                linewidth=2,
                markersize=4,
                label=style["label"],
            )
            ax.fill_between(x, lo, hi, color=style["color"], alpha=0.18, linewidth=0)

            ax.set_title(f"{ordinal[j]} Predictor", fontweight="bold")
            ax.set_xlabel("M")
            ax.set_ylabel("PBSV (RMSE)")
            ax.set_xscale("log")
            ax.axhline(0.0, color="gray", linewidth=0.8, alpha=0.5)
            ax.grid(True, which="both", alpha=0.3)

            if j == 0:
                ax.legend(loc="best")

        plt.tight_layout()
        base = os.path.join(cfg.out_dir, f"convergence_bands_{model}_{dgp_type}_{window_type}")
        plt.savefig(base + ".png", dpi=300, bbox_inches="tight")
        plt.savefig(base + ".pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"  saved: {base}.png/.pdf")


# =============================================================================
# Main
# =============================================================================
def run_one_experiment(dgp_type: str, window_type: str, *, cfg: Config) -> None:
    raw_long, summary = run_convergence_with_bands(dgp_type, window_type, cfg=cfg)

    # Save tables so you can regenerate plots without rerunning anatomy
    raw_path = os.path.join(cfg.out_dir, f"raw_pbsv_{dgp_type}_{window_type}.csv")
    sum_path = os.path.join(cfg.out_dir, f"summary_pbsv_{dgp_type}_{window_type}.csv")
    raw_long.to_csv(raw_path, index=False)
    summary.to_csv(sum_path, index=False)
    print(f"  saved: {raw_path}")
    print(f"  saved: {sum_path}")

    # Plots
    plot_overlay_with_bands(summary, dgp_type, window_type, cfg=cfg)
    plot_per_model_with_bands(summary, dgp_type, window_type, cfg=cfg)


def main():
    cfg = parse_args(CFG)
    ensure_dir(cfg.out_dir)

    experiments = [("friedman1", "rolling")]
    if cfg.run_all_experiments:
        for dgp_type in ["friedman1", "polynomial", "threshold"]:
            for window_type in ["rolling", "expanding"]:
                if not (dgp_type == "friedman1" and window_type == "rolling"):
                    experiments.append((dgp_type, window_type))

    for dgp_type, window_type in tqdm(experiments, desc="Experiments", unit="exp"):
        run_one_experiment(dgp_type, window_type, cfg=cfg)

    print("\nAll done.")


if __name__ == "__main__":
    main()

