"""
In-sample GPBSV vs out-of-sample GPBSV on the sim1 trap DGPs.

Purpose (referee response, R2 main comment 4): compute the "in-sample GPBSV" --
the window-by-window Shapley decomposition of the TRAINING loss (essentially
SAGE, Covert et al. 2020, with marginal removal, averaged over the model
sequence) -- and contrast it with the out-of-sample GPBSV on the trap DGPs
where x2 looks important but harms out-of-sample accuracy.

Expected pattern (the reason the paper does not adopt the in-sample GPBSV as
the MAS benchmark): in the structural-break and proxy-break designs x2
genuinely fit the pre-break training data, so its in-sample GPBSV is
loss-REDUCING (negative), while its out-of-sample GPBSV is loss-INCREASING
(positive). The comparison the referee proposes would therefore mask exactly
the phenomenon the GPBSV is designed to expose.

Definitions
-----------
- OoS GPBSV: exactly as in the paper / run_sim1_trap_dgps.py -- coalition
  predictions at each forecast origin, loss pooled over all forecast periods.
- IS GPBSV (window w): coalition predictions at training instances t in W_w,
  loss computed against the training targets, background data = W_w. Reported
  two ways: (i) "window-avg" = mean over sampled windows of the per-window
  decompositions (the literal sequence-averaged definition); (ii) "pooled" =
  one decomposition of the loss pooled across all sampled windows' instances.
  To keep compute bounded, windows and within-window instances are subsampled
  (defaults: 5 windows for expanding scenarios, 10 for rolling; <= 24
  instances per window); with P=4 predictors, n_iterations=48 with antithetic
  sampling is near-exact.

The DGP generators, model configuration, and fitting routines are copied
VERBATIM from run_sim1_trap_dgps.py (same seed offsets: SEEDS = 100..124) so
that the data and fitted model sequences match the existing sim1 evidence.
Keep them in sync with that script.

Usage
-----
  python run_sim1_insample_gpbsv.py --smoke          # ~2 min sanity check
  python run_sim1_insample_gpbsv.py                  # full run (background)
  python run_sim1_insample_gpbsv.py --out-dir PATH --seeds 25 --n-iterations 48
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

from anatomy import (
    Anatomy,
    AnatomyModel,
    AnatomyModelCombination,
    AnatomyModelOutputTransformer,
    AnatomyModelProvider,
    AnatomySubsets,
)

# =============================================================================
# Configuration (mirrors run_sim1_trap_dgps.py)
# =============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SIMULATION_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
DEFAULT_OUTPUT_ROOT = os.path.join(SIMULATION_ROOT, "outputs", "insample_gpbsv")

N_SEEDS = 25
SEED_START = 100

MODEL_NAMES = ["ols", "enet", "rf"]
FEATURES = ["x1", "x2", "x3", "x4"]
TRUE_ROLE = {"x1": "signal", "x2": "trap", "x3": "noise", "x4": "noise"}

ENET_ALPHA_GRID = [1e-4, 1e-3, 1e-2, 1e-1]
ENET_L1_RATIO = 0.5
ENET_CV_SPLITS = 3

RF_PARAMS = dict(
    n_estimators=100,
    max_depth=5,
    min_samples_split=10,
    bootstrap=True,
)


@dataclass(frozen=True)
class Scenario:
    name: str
    dgp_kind: str  # "A" structural break, "B" persistent noise, "C" proxy break
    n_total: int
    initial_window: int
    estimation_type: str  # "expanding" or "rolling"
    params: Dict[str, Any]


SCENARIOS: List[Scenario] = [
    Scenario(
        name="break_b08_expanding",
        dgp_kind="A",
        n_total=320,
        initial_window=200,
        estimation_type="expanding",
        params=dict(break_at=200, rho_x=0.6, rho_noise=0.3, beta1=1.0,
                    beta2_pre=0.8, beta2_post=0.0, sigma_eps=1.0),
    ),
    Scenario(
        name="persistent_rho099_rolling20",
        dgp_kind="B",
        n_total=420,
        initial_window=20,
        estimation_type="rolling",
        params=dict(rho_signal=0.5, rho_trap=0.99, rho_noise=0.3, beta1=1.0,
                    sigma_eps=1.0),
    ),
    Scenario(
        name="proxy_break_reverse_noisyx1_expanding",
        dgp_kind="C",
        n_total=320,
        initial_window=200,
        estimation_type="expanding",
        params=dict(break_at=200, rho_signal=0.8, rho_proxy_noise=0.5,
                    rho_noise=0.3, beta_signal=1.0, loading_pre=1.15,
                    loading_post=-0.75, sigma_x1_me=0.60, sigma_x2_me=0.20,
                    sigma_eps=1.0),
    ),
]
SCENARIO_BY_NAME = {s.name: s for s in SCENARIOS}


# =============================================================================
# DGPs (copied verbatim from run_sim1_trap_dgps.py)
# =============================================================================
def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def ar1_unitvar(n: int, rho: float, seed: int, std: float = 1.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    e = rng.normal(0.0, std, size=n)
    x = np.empty(n, dtype=float)
    x[0] = e[0]
    sc = np.sqrt(max(1e-12, 1.0 - rho ** 2))
    for t in range(1, n):
        x[t] = rho * x[t - 1] + sc * e[t]
    return x


def dgp_structural_break(n_total, break_at, initial_window, rho_x, rho_noise,
                         beta1, beta2_pre, beta2_post, sigma_eps, seed) -> pd.DataFrame:
    assert break_at == initial_window, "DGP A requires break_at == initial_window."
    x1 = ar1_unitvar(n_total, rho=rho_x, seed=seed + 1)
    x2 = ar1_unitvar(n_total, rho=rho_x, seed=seed + 2)
    x3 = ar1_unitvar(n_total, rho=rho_noise, seed=seed + 3)
    x4 = ar1_unitvar(n_total, rho=rho_noise, seed=seed + 4)
    beta2 = np.full(n_total, float(beta2_post), dtype=float)
    beta2[:break_at] = float(beta2_pre)
    rng = np.random.default_rng(seed + 99)
    eps = rng.normal(0.0, float(sigma_eps), size=n_total)
    y_next = float(beta1) * x1[:-1] + beta2[:-1] * x2[:-1] + eps[1:]
    df = pd.DataFrame({"x1": x1[:-1], "x2": x2[:-1], "x3": x3[:-1], "x4": x4[:-1], "y": y_next})
    df.index = pd.date_range("2000-01-01", periods=len(df), freq="M").map(lambda d: d.date())
    return df


def dgp_persistent_noise(n_total, rho_signal, rho_trap, rho_noise, beta1,
                         sigma_eps, seed) -> pd.DataFrame:
    x1 = ar1_unitvar(n_total, rho=rho_signal, seed=seed + 1)
    x2 = ar1_unitvar(n_total, rho=rho_trap, seed=seed + 2)
    x3 = ar1_unitvar(n_total, rho=rho_noise, seed=seed + 3)
    x4 = ar1_unitvar(n_total, rho=rho_noise, seed=seed + 4)
    rng = np.random.default_rng(seed + 99)
    eps = rng.normal(0.0, float(sigma_eps), size=n_total)
    y_next = float(beta1) * x1[:-1] + eps[1:]
    df = pd.DataFrame({"x1": x1[:-1], "x2": x2[:-1], "x3": x3[:-1], "x4": x4[:-1], "y": y_next})
    df.index = pd.date_range("2000-01-01", periods=len(df), freq="M").map(lambda d: d.date())
    return df


def dgp_proxy_breakdown(n_total, break_at, rho_signal, rho_proxy_noise, rho_noise,
                        beta_signal, loading_pre, loading_post, sigma_x1_me,
                        sigma_x2_me, sigma_eps, seed) -> pd.DataFrame:
    if not (0 < break_at < n_total):
        raise ValueError("break_at must lie strictly inside the sample.")
    s = ar1_unitvar(n_total, rho_signal, seed + 1)
    proxy_noise = ar1_unitvar(n_total, rho_proxy_noise, seed + 2)
    x3 = ar1_unitvar(n_total, rho_noise, seed + 3)
    x4 = ar1_unitvar(n_total, rho_noise, seed + 4)
    rng = np.random.default_rng(seed + 99)
    me1 = rng.normal(0.0, float(sigma_x1_me), size=n_total)
    eps = rng.normal(0.0, float(sigma_eps), size=n_total)
    loading = np.full(n_total, float(loading_post), dtype=float)
    loading[:break_at] = float(loading_pre)
    x1 = s + me1
    x2 = loading * s + float(sigma_x2_me) * proxy_noise
    y_next = float(beta_signal) * s[:-1] + eps[1:]
    df = pd.DataFrame({"x1": x1[:-1], "x2": x2[:-1], "x3": x3[:-1], "x4": x4[:-1], "y": y_next})
    df.index = pd.date_range("2000-01-01", periods=len(df), freq="M").map(lambda d: d.date())
    return df


def make_xy(scenario: Scenario, seed: int) -> pd.DataFrame:
    if scenario.dgp_kind == "A":
        return dgp_structural_break(
            n_total=scenario.n_total, initial_window=scenario.initial_window,
            seed=seed, **scenario.params)
    if scenario.dgp_kind == "B":
        return dgp_persistent_noise(n_total=scenario.n_total, seed=seed, **scenario.params)
    if scenario.dgp_kind == "C":
        return dgp_proxy_breakdown(n_total=scenario.n_total, seed=seed, **scenario.params)
    raise ValueError(f"Unknown dgp_kind: {scenario.dgp_kind}")


# =============================================================================
# Models (copied verbatim from run_sim1_trap_dgps.py)
# =============================================================================
def select_enet_alpha_once(X: pd.DataFrame, y: pd.Series, seed: int) -> float:
    cv = TimeSeriesSplit(n_splits=int(ENET_CV_SPLITS))
    best_alpha = float(ENET_ALPHA_GRID[0])
    best_score = -np.inf
    for a in ENET_ALPHA_GRID:
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("enet", ElasticNet(alpha=float(a), l1_ratio=float(ENET_L1_RATIO),
                                max_iter=5000, random_state=int(seed))),
        ])
        score = cross_val_score(pipe, X, y, cv=cv, scoring="neg_mean_squared_error").mean()
        if score > best_score:
            best_score = float(score)
            best_alpha = float(a)
    return best_alpha


def fit_model(model_name: str, X: pd.DataFrame, y: pd.Series, *, seed_base: int,
              period: int, enet_alpha: Optional[float]) -> Any:
    if model_name == "ols":
        return LinearRegression().fit(X, y)
    if model_name == "enet":
        if enet_alpha is None:
            raise ValueError("enet_alpha must be provided for ENet.")
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("enet", ElasticNet(alpha=float(enet_alpha), l1_ratio=float(ENET_L1_RATIO),
                                max_iter=5000, random_state=int(seed_base + 10_000 + period))),
        ])
        return pipe.fit(X, y)
    if model_name == "rf":
        rf = RandomForestRegressor(**RF_PARAMS, random_state=int(seed_base + 20_000 + period))
        return rf.fit(X, y)
    raise ValueError(f"Unknown model: {model_name}")


def estimator_to_anatomy_model(estimator: Any, feature_names: List[str]) -> AnatomyModel:
    def pred_fn(xs: np.ndarray) -> np.ndarray:
        Xdf = pd.DataFrame(xs, columns=feature_names)
        return np.asarray(estimator.predict(Xdf)).ravel()
    return AnatomyModel(pred_fn)


# =============================================================================
# Transformers
# =============================================================================
MSE_TRANSFORMER = AnatomyModelOutputTransformer(
    transform=lambda y_hat, y: float(np.mean((y - y_hat) ** 2)))
RMSE_TRANSFORMER = AnatomyModelOutputTransformer(
    transform=lambda y_hat, y: float(np.sqrt(np.mean((y - y_hat) ** 2))))


def _as_row(x):
    return x.iloc[0] if isinstance(x, pd.DataFrame) else x


def explain_loss(anatomy_obj: Anatomy, model_key: str, transformer,
                 subset: Optional[pd.Index] = None) -> pd.Series:
    """Global loss decomposition for one model; returns base + per-feature phis."""
    df = anatomy_obj.explain(
        model_sets=AnatomyModelCombination(groups={model_key: [model_key]}),
        transformer=transformer,
        explanation_subset=subset,
    )
    return _as_row(df.loc[model_key])


# =============================================================================
# Out-of-sample anatomy (identical wiring to run_sim1_trap_dgps.run_anatomy)
# =============================================================================
def run_oos_anatomy(xy: pd.DataFrame, subsets: AnatomySubsets, n_iterations: int,
                    save_path: str, *, seed_base: int, enet_alpha: float,
                    n_jobs: int) -> Anatomy:
    feats = [c for c in xy.columns if c != "y"]

    def mapper(key: AnatomyModelProvider.PeriodKey) -> AnatomyModelProvider.PeriodValue:
        train = xy.iloc[subsets.get_train_subset(key.period)]
        test = xy.iloc[subsets.get_test_subset(key.period)]
        est = fit_model(key.model_name, train[feats], train["y"],
                        seed_base=seed_base, period=key.period, enet_alpha=enet_alpha)
        return AnatomyModelProvider.PeriodValue(train, test, estimator_to_anatomy_model(est, feats))

    provider = AnatomyModelProvider(
        n_periods=subsets.n_periods, n_features=len(feats),
        model_names=list(MODEL_NAMES), y_name="y", provider_fn=mapper)

    np.random.seed(seed_base + 777)  # permutation draws use the global RNG
    Anatomy(provider=provider, n_iterations=int(n_iterations)).precompute(
        n_jobs=n_jobs, save_path=save_path)
    return Anatomy.load(save_path)


# =============================================================================
# In-sample anatomy: pseudo-periods = sampled windows; test = sampled TRAIN rows
# =============================================================================
def sample_evenly(n: int, k: int) -> List[int]:
    if k >= n:
        return list(range(n))
    pts = np.linspace(0, n - 1, num=k)
    return sorted(set(int(round(p)) for p in pts))


def run_is_anatomy(xy: pd.DataFrame, subsets: AnatomySubsets, n_iterations: int,
                   save_path: str, *, seed_base: int, enet_alpha: float,
                   n_windows: int, n_instances: int, n_jobs: int):
    """
    Returns (anatomy_obj, window_row_labels) where window_row_labels maps each
    sampled window id -> the unique index labels of its in-sample instances
    (used for per-window explanation subsets).
    """
    feats = [c for c in xy.columns if c != "y"]
    window_ids = sample_evenly(subsets.n_periods, n_windows)

    # Pre-build per-window test frames with globally unique index labels.
    window_tests: Dict[int, pd.DataFrame] = {}
    window_row_labels: Dict[int, List[str]] = {}
    for w in window_ids:
        train = xy.iloc[subsets.get_train_subset(w)]
        rows = sample_evenly(len(train), n_instances)
        test = train.iloc[rows].copy()
        labels = [f"w{w:04d}|{str(ix)}" for ix in test.index]
        test.index = pd.Index(labels, name=xy.index.name)
        window_tests[w] = test
        window_row_labels[w] = labels

    def mapper(key: AnatomyModelProvider.PeriodKey) -> AnatomyModelProvider.PeriodValue:
        w = window_ids[key.period]
        train = xy.iloc[subsets.get_train_subset(w)]
        est = fit_model(key.model_name, train[feats], train["y"],
                        seed_base=seed_base, period=w, enet_alpha=enet_alpha)
        return AnatomyModelProvider.PeriodValue(
            train, window_tests[w], estimator_to_anatomy_model(est, feats))

    provider = AnatomyModelProvider(
        n_periods=len(window_ids), n_features=len(feats),
        model_names=list(MODEL_NAMES), y_name="y", provider_fn=mapper)

    np.random.seed(seed_base + 778)
    Anatomy(provider=provider, n_iterations=int(n_iterations)).precompute(
        n_jobs=n_jobs, save_path=save_path)
    return Anatomy.load(save_path), window_row_labels


# =============================================================================
# Per-seed computation
# =============================================================================
def run_seed(scenario: Scenario, seed: int, *, n_iterations: int, n_windows: int,
             n_instances: int, n_jobs: int, tmp_dir: str) -> List[Dict[str, Any]]:
    xy = make_xy(scenario, seed)
    est_type = (AnatomySubsets.EstimationType.EXPANDING
                if scenario.estimation_type == "expanding"
                else AnatomySubsets.EstimationType.ROLLING)
    subsets = AnatomySubsets.generate(
        index=xy.index, initial_window=scenario.initial_window,
        estimation_type=est_type, periods=1, gap=0)

    feats = [c for c in xy.columns if c != "y"]
    first_train = xy.iloc[subsets.get_train_subset(0)]
    enet_alpha = select_enet_alpha_once(first_train[feats], first_train["y"], seed)

    oos_bin = os.path.join(tmp_dir, f"oos_{scenario.name}_s{seed}.bin")
    is_bin = os.path.join(tmp_dir, f"is_{scenario.name}_s{seed}.bin")
    for p in (oos_bin, is_bin):
        if os.path.exists(p):
            os.remove(p)

    t0 = time.time()
    oos = run_oos_anatomy(xy, subsets, n_iterations, oos_bin,
                          seed_base=seed, enet_alpha=enet_alpha, n_jobs=n_jobs)
    t1 = time.time()
    isa, window_rows = run_is_anatomy(xy, subsets, n_iterations, is_bin,
                                      seed_base=seed, enet_alpha=enet_alpha,
                                      n_windows=n_windows, n_instances=n_instances,
                                      n_jobs=n_jobs)
    t2 = time.time()

    records: List[Dict[str, Any]] = []
    for model in MODEL_NAMES:
        oos_mse = explain_loss(oos, model, MSE_TRANSFORMER)
        oos_rmse = explain_loss(oos, model, RMSE_TRANSFORMER)

        # pooled IS decomposition (all sampled instances at once)
        is_mse_pooled = explain_loss(isa, model, MSE_TRANSFORMER)
        is_rmse_pooled = explain_loss(isa, model, RMSE_TRANSFORMER)

        # window-avg IS decomposition (the literal sequence-averaged definition)
        per_window_mse = []
        per_window_rmse = []
        for w, labels in window_rows.items():
            sub = pd.Index(labels)
            per_window_mse.append(explain_loss(isa, model, MSE_TRANSFORMER, subset=sub))
            per_window_rmse.append(explain_loss(isa, model, RMSE_TRANSFORMER, subset=sub))
        is_mse_avg = pd.concat(per_window_mse, axis=1).mean(axis=1)
        is_rmse_avg = pd.concat(per_window_rmse, axis=1).mean(axis=1)

        for feat in feats:
            records.append(dict(
                scenario=scenario.name, seed=seed, model=model, feature=feat,
                role=TRUE_ROLE[feat],
                is_gpbsv_mse=float(is_mse_avg[feat]),
                is_gpbsv_mse_pooled=float(is_mse_pooled[feat]),
                oos_gpbsv_mse=float(oos_mse[feat]),
                is_gpbsv_rmse=float(is_rmse_avg[feat]),
                is_gpbsv_rmse_pooled=float(is_rmse_pooled[feat]),
                oos_gpbsv_rmse=float(oos_rmse[feat]),
                is_base_mse=float(is_mse_pooled["base_contribution"]),
                oos_base_mse=float(oos_mse["base_contribution"]),
                n_windows_sampled=len(window_rows),
                n_instances_per_window=n_instances,
                n_iterations=n_iterations,
                sec_oos=round(t1 - t0, 1), sec_is=round(t2 - t1, 1),
            ))

    for p in (oos_bin, is_bin):
        if os.path.exists(p):
            os.remove(p)
    return records


# =============================================================================
# Summaries and figure
# =============================================================================
def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (scen, model, feat), g in df.groupby(["scenario", "model", "feature"]):
        rows.append(dict(
            scenario=scen, model=model, feature=feat, role=TRUE_ROLE[feat],
            n_seeds=len(g),
            mean_is_gpbsv_mse=g["is_gpbsv_mse"].mean(),
            mean_oos_gpbsv_mse=g["oos_gpbsv_mse"].mean(),
            std_is_gpbsv_mse=g["is_gpbsv_mse"].std(),
            std_oos_gpbsv_mse=g["oos_gpbsv_mse"].std(),
            share_is_beneficial=float((g["is_gpbsv_mse"] < 0).mean()),
            share_oos_harmful=float((g["oos_gpbsv_mse"] > 0).mean()),
            share_trap_signature=float(
                ((g["is_gpbsv_mse"] < 0) & (g["oos_gpbsv_mse"] > 0)).mean()),
        ))
    return pd.DataFrame(rows)


# Visual encoding identical to postprocess_sim1_referee.py (the figure set the
# referee-response material already uses): per-feature markers + tab10 colors,
# no text labels on points, frameless top-center legend.
from matplotlib.lines import Line2D

FEATURE_MARKERS = {"x1": "o", "x2": "s", "x3": "^", "x4": "D"}
_cmap = plt.get_cmap("tab10")
FEATURE_COLORS = {"x1": _cmap(0), "x2": _cmap(1), "x3": _cmap(2), "x4": _cmap(3)}


def _feature_legend_handles():
    return [
        Line2D([0], [0], marker=FEATURE_MARKERS[f], color="none",
               markerfacecolor=FEATURE_COLORS[f], markeredgecolor="none",
               markersize=7, linestyle="None", label=f)
        for f in FEATURES
    ]


def _styled_panels(scenario: str, title_tail: str):
    ncols = len(MODEL_NAMES)
    fig, axes = plt.subplots(1, ncols, figsize=(4.25 * ncols, 4.0), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, model in zip(axes, MODEL_NAMES):
        ax.axhline(0.0, linewidth=1.0, alpha=0.6)
        ax.axvline(0.0, linewidth=1.0, alpha=0.6)
        ax.set_title(model.upper())
        ax.grid(True, alpha=0.2)
    fig.suptitle(f"{scenario} — {title_tail}", y=1.02)
    fig.legend(handles=_feature_legend_handles(), loc="upper center",
               ncol=len(FEATURES), frameon=False, bbox_to_anchor=(0.5, 1.10))
    return fig, axes


def _save(fig, out_path_no_ext: str) -> None:
    fig.tight_layout()
    fig.savefig(f"{out_path_no_ext}.pdf", bbox_inches="tight")
    fig.savefig(f"{out_path_no_ext}.png", bbox_inches="tight", dpi=150)
    plt.close(fig)


def make_figure(df: pd.DataFrame, scenario: str, out_path_no_ext: str) -> None:
    """Per-seed scatter + averages-with-error-bars versions, styled like the
    fig_importance_vs_gpbsv / fig_importance_vs_gpbsv_avg pair."""
    sub = df[df["scenario"] == scenario]

    # Per-seed version (style of make_importance_vs_gpbsv_pdf)
    fig, axes = _styled_panels(scenario, "in-sample vs out-of-sample GPBSV")
    for ax, model in zip(axes, MODEL_NAMES):
        m = sub[sub["model"] == model]
        for f in FEATURES:
            d = m[m["feature"] == f]
            if d.empty:
                continue
            ax.scatter(d["is_gpbsv_mse"], d["oos_gpbsv_mse"], s=22, alpha=0.55,
                       marker=FEATURE_MARKERS[f], c=[FEATURE_COLORS[f]],
                       edgecolors="none")
        ax.set_xlabel("In-sample GPBSV (ΔMSE)")
    axes[0].set_ylabel("Out-of-sample GPBSV (ΔMSE)")
    _save(fig, out_path_no_ext)

    # Averages version (style of make_importance_vs_gpbsv_avg_pdf)
    fig, axes = _styled_panels(scenario, "in-sample vs out-of-sample GPBSV — averages across seeds")
    for ax, model in zip(axes, MODEL_NAMES):
        m = sub[sub["model"] == model]
        for f in FEATURES:
            d = m[m["feature"] == f]
            if d.empty:
                continue
            x, y = d["is_gpbsv_mse"], d["oos_gpbsv_mse"]
            xerr = float(x.std()) if np.isfinite(x.std()) else 0.0
            yerr = float(y.std()) if np.isfinite(y.std()) else 0.0
            ax.errorbar([float(x.mean())], [float(y.mean())], xerr=[xerr], yerr=[yerr],
                        fmt=FEATURE_MARKERS[f], markersize=7, capsize=3, alpha=0.9,
                        color=FEATURE_COLORS[f], markerfacecolor=FEATURE_COLORS[f],
                        markeredgecolor="none")
        ax.set_xlabel("Mean in-sample GPBSV (ΔMSE)")
    axes[0].set_ylabel("Mean out-of-sample GPBSV (ΔMSE)")
    _save(fig, f"{out_path_no_ext}_avg")


# =============================================================================
# Main
# =============================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="In-sample vs out-of-sample GPBSV on the sim1 trap DGPs.")
    p.add_argument("--out-dir", default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--scenarios", nargs="+", default=[s.name for s in SCENARIOS],
                   choices=list(SCENARIO_BY_NAME))
    p.add_argument("--seeds", type=int, default=N_SEEDS, help="number of MC seeds (from 100)")
    p.add_argument("--n-iterations", type=int, default=48,
                   help="permutation draws (antithetic doubles them); P=4 so 48 is near-exact")
    p.add_argument("--n-windows", type=int, default=5,
                   help="sampled training windows for the IS side (expanding scenarios)")
    p.add_argument("--n-windows-rolling", type=int, default=10,
                   help="sampled training windows for the IS side (rolling scenarios)")
    p.add_argument("--n-instances", type=int, default=24,
                   help="sampled in-sample instances per window")
    p.add_argument("--jobs", type=int, default=max(1, min(os.cpu_count() or 1, 8)))
    p.add_argument("--smoke", action="store_true",
                   help="tiny run: first scenario, 2 seeds, 8 iterations, 2 windows, 8 instances")
    p.add_argument("--replot", action="store_true",
                   help="regenerate figures from the saved per-seed CSVs; no computation")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    scenarios = [SCENARIO_BY_NAME[s] for s in args.scenarios]

    if args.replot:
        for scenario in scenarios:
            scen_dir = os.path.join(args.out_dir, scenario.name)
            csv_path = os.path.join(scen_dir, "is_vs_oos_gpbsv_per_seed.csv")
            if not os.path.exists(csv_path):
                print(f"skip {scenario.name}: no {csv_path}")
                continue
            scen_df = pd.read_csv(csv_path)
            make_figure(scen_df, scenario.name, os.path.join(scen_dir, "fig_is_vs_oos_gpbsv"))
            print(f"replotted {scenario.name}")
        return

    seeds = list(range(SEED_START, SEED_START + args.seeds))
    n_iterations, n_instances = args.n_iterations, args.n_instances

    if args.smoke:
        scenarios = [SCENARIO_BY_NAME["break_b08_expanding"]]
        seeds = seeds[:2]
        n_iterations, n_instances = 8, 8

    out_dir = ensure_dir(args.out_dir)
    tmp_dir = ensure_dir(os.path.join(out_dir, "_tmp_bins"))

    all_records: List[Dict[str, Any]] = []
    t_start = time.time()
    for scenario in scenarios:
        n_windows = (args.n_windows if scenario.estimation_type == "expanding"
                     else args.n_windows_rolling)
        if args.smoke:
            n_windows = 2
        scen_dir = ensure_dir(os.path.join(out_dir, scenario.name))
        for i, seed in enumerate(seeds):
            recs = run_seed(scenario, seed, n_iterations=n_iterations,
                            n_windows=n_windows, n_instances=n_instances,
                            n_jobs=args.jobs, tmp_dir=tmp_dir)
            all_records.extend(recs)
            el = time.time() - t_start
            print(f"[{scenario.name}] seed {seed} done "
                  f"({i + 1}/{len(seeds)}; elapsed {el / 60:.1f} min)", flush=True)

        scen_df = pd.DataFrame([r for r in all_records if r["scenario"] == scenario.name])
        scen_df.to_csv(os.path.join(scen_dir, "is_vs_oos_gpbsv_per_seed.csv"), index=False)
        summarize(scen_df).to_csv(os.path.join(scen_dir, "summary.csv"), index=False)
        make_figure(scen_df, scenario.name, os.path.join(scen_dir, "fig_is_vs_oos_gpbsv"))

    full = pd.DataFrame(all_records)
    full.to_csv(os.path.join(out_dir, "is_vs_oos_gpbsv_all.csv"), index=False)
    summary = summarize(full)
    summary.to_csv(os.path.join(out_dir, "summary_all.csv"), index=False)

    print("\n=== Trap signature (x2): share of seeds with IS-GPBSV < 0 AND OoS-GPBSV > 0 ===")
    x2 = summary[summary["feature"] == "x2"][
        ["scenario", "model", "mean_is_gpbsv_mse", "mean_oos_gpbsv_mse",
         "share_is_beneficial", "share_oos_harmful", "share_trap_signature"]]
    print(x2.to_string(index=False), flush=True)
    print(f"\nTotal wall time: {(time.time() - t_start) / 60:.1f} min")
    print(f"Outputs in: {out_dir}")


if __name__ == "__main__":
    main()
