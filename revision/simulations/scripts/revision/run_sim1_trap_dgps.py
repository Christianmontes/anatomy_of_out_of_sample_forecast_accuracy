"""
===============================================================================
Simulation 1 — Final Master Script for Paper Revision (Console-first)
===============================================================================

Purpose
-------
Produce a controlled “smoking gun” simulation showing a predictor x2 that:
  • looks important by classical measures (|t|, standardized beta, APE, perm VI, PDP range),
  • looks important by RF-native measures (OOB VI, RF APE),
  • looks important by prediction-based Shapley importance (oShapley-VI),
BUT
  • harms out-of-sample accuracy (GPBSV > 0),
  • and dropping x2 improves OOS MSE.

Design (per prompt; NO grid search)
-----------------------------------
DGP A (structural break), expanding windows:
  break_at = initial_window = 200
  beta_pre = 0.8  (headline)
  beta_pre = 1.5  (appendix)

DGP B (persistent noise / spurious regression), rolling windows:
  rho_trap = 0.99, initial_window = 20, N = 420

DGP C (proxy breakdown), expanding windows:
  x2 is an early-sample proxy for a latent signal, then the proxy relation
  reverses after the break.

Models
------
  - OLS
  - Elastic Net (alpha chosen once per seed via TimeSeriesSplit CV; Pipeline avoids leakage)
  - Random Forest (100 trees, max_depth=5, min_samples_split=10)

Key implementation principle
----------------------------
One unified fit_model() is used everywhere: anatomy, classical measures, PDP, perm VI, APE, and OOS refits.

Outputs
-------
Console:
  - Per-seed status line (per model)
  - Success rates and averages (per DGP x model)
  - Representative exhibit table (per DGP x model)

Disk:
  - per-scenario grand_summary.csv
  - per-seed per-model comparison tables (.csv)

Dependencies:
  anatomy==0.1.6, numpy, pandas, scikit-learn, joblib

Run:
  python run_sim1_trap_dgps.py
"""

import argparse
import os
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from anatomy import (
    Anatomy,
    AnatomyModel,
    AnatomyModelCombination,
    AnatomyModelOutputTransformer,
    AnatomyModelProvider,
    AnatomySubsets,
    MAS,
)

from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance as sk_perm_importance
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# Quiet joblib verbosity (anatomy uses joblib internally)
import joblib
_original_parallel_call = joblib.Parallel.__call__
def _quiet_parallel_call(self, *args, **kwargs):
    self.verbose = 0
    return _original_parallel_call(self, *args, **kwargs)
joblib.Parallel.__call__ = _quiet_parallel_call


# =============================================================================
# Global configuration
# =============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SIMULATION_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
OUTPUT_ROOT = os.path.join(SIMULATION_ROOT, "outputs", "sim1_master_out")
CLEANUP_ANATOMY_BIN = True

N_SEEDS = 25
SEEDS = list(range(100, 100 + N_SEEDS))

MODEL_NAMES = ["ols", "enet", "rf"]

N_ITERATIONS_SHAPLEY = 400

# Snapshot settings for expensive “classical” proxies
N_SNAPSHOTS = 5
PERM_REPEATS_TRAIN = 15
PDP_GRID_SIZE = 21

# Thresholds
EPS = 0.02                 # “harmful/beneficial/near-zero” threshold for GPBSV
IMPROVE_PCT_MIN = 10.0     # “substantial” improvement threshold for smoking-gun flags

FEATURES = ["x1", "x2", "x3", "x4"]
TRUE_ROLE = {"x1": "signal", "x2": "trap", "x3": "noise", "x4": "noise"}

# Elastic Net configuration
ENET_ALPHA_GRID = [1e-4, 1e-3, 1e-2, 1e-1]
ENET_L1_RATIO = 0.5
ENET_CV_SPLITS = 3

# Random Forest configuration (as requested)
RF_PARAMS = dict(
    n_estimators=100,
    max_depth=5,
    min_samples_split=10,
    bootstrap=True,
)
RF_OOB_PERM_REPEATS = 2
RF_APE_DELTA_SCALE = 0.1
RF_APE_MAX_OBS = 300


# =============================================================================
# Scenario specification
# =============================================================================
@dataclass(frozen=True)
class Scenario:
    name: str
    dgp_kind: str  # "A" structural break, "B" persistent noise, "C" proxy break
    n_total: int
    initial_window: int
    estimation_type: str  # "expanding" or "rolling"
    params: Dict[str, Any]


SCENARIOS: List[Scenario] = [
    # DGP A headline
    Scenario(
        name="break_b08_expanding",
        dgp_kind="A",
        n_total=320,
        initial_window=200,
        estimation_type="expanding",
        params=dict(
            break_at=200,
            rho_x=0.6,
            rho_noise=0.3,
            beta1=1.0,
            beta2_pre=0.8,
            beta2_post=0.0,
            sigma_eps=1.0,
        ),
    ),
    # DGP A appendix
    Scenario(
        name="break_b15_expanding",
        dgp_kind="A",
        n_total=320,
        initial_window=200,
        estimation_type="expanding",
        params=dict(
            break_at=200,
            rho_x=0.6,
            rho_noise=0.3,
            beta1=1.0,
            beta2_pre=1.5,
            beta2_post=0.0,
            sigma_eps=1.0,
        ),
    ),
    # DGP B persistent noise
    Scenario(
        name="persistent_rho099_rolling20",
        dgp_kind="B",
        n_total=420,
        initial_window=20,
        estimation_type="rolling",
        params=dict(
            rho_signal=0.5,
            rho_trap=0.99,
            rho_noise=0.3,
            beta1=1.0,
            sigma_eps=1.0,
        ),
    ),
    # DGP C proxy breakdown
    Scenario(
        name="proxy_break_reverse_noisyx1_expanding",
        dgp_kind="C",
        n_total=320,
        initial_window=200,
        estimation_type="expanding",
        params=dict(
            break_at=200,
            rho_signal=0.8,
            rho_proxy_noise=0.5,
            rho_noise=0.3,
            beta_signal=1.0,
            loading_pre=1.15,
            loading_post=-0.75,
            sigma_x1_me=0.60,
            sigma_x2_me=0.20,
            sigma_eps=1.0,
        ),
    ),
]


# =============================================================================
# Utilities and DGPs
# =============================================================================
def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def ar1_unitvar(n: int, rho: float, seed: int, std: float = 1.0) -> np.ndarray:
    """
    Stationary AR(1) with ~unit unconditional variance:
      x_t = rho x_{t-1} + sqrt(1-rho^2) e_t, e_t ~ N(0, std^2)
    """
    rng = np.random.default_rng(seed)
    e = rng.normal(0.0, std, size=n)
    x = np.empty(n, dtype=float)
    x[0] = e[0]
    sc = np.sqrt(max(1e-12, 1.0 - rho ** 2))
    for t in range(1, n):
        x[t] = rho * x[t - 1] + sc * e[t]
    return x


def dgp_structural_break(
    n_total: int,
    break_at: int,
    initial_window: int,
    rho_x: float,
    rho_noise: float,
    beta1: float,
    beta2_pre: float,
    beta2_post: float,
    sigma_eps: float,
    seed: int,
) -> pd.DataFrame:
    """
    y_{t+1} = beta1 * x1_t + beta2(t) * x2_t + eps_{t+1}
    beta2(t) = beta2_pre for t < break_at, beta2_post for t >= break_at.
    """
    assert break_at == initial_window, "DGP A requires break_at == initial_window for clean alignment."

    x1 = ar1_unitvar(n_total, rho=rho_x, seed=seed + 1)
    x2 = ar1_unitvar(n_total, rho=rho_x, seed=seed + 2)
    x3 = ar1_unitvar(n_total, rho=rho_noise, seed=seed + 3)
    x4 = ar1_unitvar(n_total, rho=rho_noise, seed=seed + 4)

    beta2 = np.full(n_total, float(beta2_post), dtype=float)
    beta2[:break_at] = float(beta2_pre)

    rng = np.random.default_rng(seed + 99)
    eps = rng.normal(0.0, float(sigma_eps), size=n_total)

    y_next = float(beta1) * x1[:-1] + beta2[:-1] * x2[:-1] + eps[1:]

    df = pd.DataFrame(
        {"x1": x1[:-1], "x2": x2[:-1], "x3": x3[:-1], "x4": x4[:-1], "y": y_next}
    )
    df.index = pd.date_range("2000-01-01", periods=len(df), freq="M").map(lambda d: d.date())
    return df


def dgp_persistent_noise(
    n_total: int,
    rho_signal: float,
    rho_trap: float,
    rho_noise: float,
    beta1: float,
    sigma_eps: float,
    seed: int,
) -> pd.DataFrame:
    """
    y_{t+1} = beta1 * x1_t + eps_{t+1}
    x2 is near-unit-root AR(1) with no predictive power.
    """
    x1 = ar1_unitvar(n_total, rho=rho_signal, seed=seed + 1)
    x2 = ar1_unitvar(n_total, rho=rho_trap, seed=seed + 2)
    x3 = ar1_unitvar(n_total, rho=rho_noise, seed=seed + 3)
    x4 = ar1_unitvar(n_total, rho=rho_noise, seed=seed + 4)

    rng = np.random.default_rng(seed + 99)
    eps = rng.normal(0.0, float(sigma_eps), size=n_total)

    y_next = float(beta1) * x1[:-1] + eps[1:]

    df = pd.DataFrame(
        {"x1": x1[:-1], "x2": x2[:-1], "x3": x3[:-1], "x4": x4[:-1], "y": y_next}
    )
    df.index = pd.date_range("2000-01-01", periods=len(df), freq="M").map(lambda d: d.date())
    return df


def dgp_proxy_breakdown(
    n_total: int,
    break_at: int,
    rho_signal: float,
    rho_proxy_noise: float,
    rho_noise: float,
    beta_signal: float,
    loading_pre: float,
    loading_post: float,
    sigma_x1_me: float,
    sigma_x2_me: float,
    sigma_eps: float,
    seed: int,
) -> pd.DataFrame:
    """
    y_{t+1} = beta_signal * s_t + eps_{t+1}
    x1_t = s_t + measurement error
    x2_t = loading_t * s_t + proxy noise

    x2 has no direct structural effect on y. It is an early proxy for the
    latent signal, then becomes a stale/reversed proxy after break_at.
    """
    if break_at != int(break_at):
        raise ValueError("break_at must be an integer.")
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

    df = pd.DataFrame(
        {"x1": x1[:-1], "x2": x2[:-1], "x3": x3[:-1], "x4": x4[:-1], "y": y_next}
    )
    df.index = pd.date_range("2000-01-01", periods=len(df), freq="M").map(lambda d: d.date())
    return df


def snapshot_periods(n_periods: int, n_snapshots: int = 5) -> List[int]:
    """
    Evenly spaced snapshot periods for expensive measures.
    """
    if n_periods <= 0:
        return [0]
    if n_snapshots <= 1:
        return [0]
    pts = np.linspace(0, max(0, n_periods - 1), num=n_snapshots)
    out = sorted(set(int(round(x)) for x in pts))
    return [p for p in out if 0 <= p < n_periods]


def rank_desc(s: pd.Series) -> pd.Series:
    """
    Rank with 1 = largest (most important).
    """
    return (-s).rank(method="min").astype(int)


def interpret_pbsv(v: float) -> str:
    if v < -EPS:
        return "beneficial"
    if v > EPS:
        return "harmful"
    return "near-zero"


# =============================================================================
# Model training (unified)
# =============================================================================
def select_enet_alpha_once(X: pd.DataFrame, y: pd.Series, seed: int) -> float:
    """
    Choose ENet alpha on the initial training window using Pipeline + TimeSeriesSplit.
    This avoids leakage from scaling during CV.
    """
    cv = TimeSeriesSplit(n_splits=int(ENET_CV_SPLITS))
    best_alpha = float(ENET_ALPHA_GRID[0])
    best_score = -np.inf

    for a in ENET_ALPHA_GRID:
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("enet", ElasticNet(alpha=float(a), l1_ratio=float(ENET_L1_RATIO),
                                    max_iter=5000, random_state=int(seed))),
            ]
        )
        score = cross_val_score(pipe, X, y, cv=cv, scoring="neg_mean_squared_error").mean()
        if score > best_score:
            best_score = float(score)
            best_alpha = float(a)

    return best_alpha


def fit_model(
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    seed_base: int,
    period: int,
    enet_alpha: Optional[float],
) -> Any:
    """
    Unified training routine used everywhere.
    """
    if model_name == "ols":
        return LinearRegression().fit(X, y)

    if model_name == "enet":
        if enet_alpha is None:
            raise ValueError("enet_alpha must be provided for ENet.")
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("enet", ElasticNet(alpha=float(enet_alpha), l1_ratio=float(ENET_L1_RATIO),
                                    max_iter=5000, random_state=int(seed_base + 10_000 + period))),
            ]
        )
        return pipe.fit(X, y)

    if model_name == "rf":
        rf = RandomForestRegressor(
            **RF_PARAMS,
            random_state=int(seed_base + 20_000 + period),
        )
        return rf.fit(X, y)

    raise ValueError(f"Unknown model: {model_name}")


def estimator_to_anatomy_model(estimator: Any, feature_names: List[str]) -> AnatomyModel:
    def pred_fn(xs: np.ndarray) -> np.ndarray:
        Xdf = pd.DataFrame(xs, columns=feature_names)
        return np.asarray(estimator.predict(Xdf)).ravel()
    return AnatomyModel(pred_fn)


# =============================================================================
# Anatomy (oShapley-VI, GPBSV, MAS)
# =============================================================================
def run_anatomy(
    xy: pd.DataFrame,
    subsets: AnatomySubsets,
    model_names: Sequence[str],
    n_iterations: int,
    save_path: str,
    *,
    seed_base: int,
    enet_alpha: float,
) -> Anatomy:
    feats = [c for c in xy.columns if c != "y"]

    def mapper(key: AnatomyModelProvider.PeriodKey) -> AnatomyModelProvider.PeriodValue:
        train = xy.iloc[subsets.get_train_subset(key.period)]
        test = xy.iloc[subsets.get_test_subset(key.period)]
        Xtr, ytr = train[feats], train["y"]

        est = fit_model(
            key.model_name,
            Xtr,
            ytr,
            seed_base=seed_base,
            period=key.period,
            enet_alpha=enet_alpha,
        )
        model = estimator_to_anatomy_model(est, feats)
        return AnatomyModelProvider.PeriodValue(train, test, model)

    provider = AnatomyModelProvider(
        n_periods=subsets.n_periods,
        n_features=len(feats),
        model_names=list(model_names),
        y_name="y",
        provider_fn=mapper,
    )

    n_jobs = max(1, min(os.cpu_count() or 1, 8))
    Anatomy(provider=provider, n_iterations=int(n_iterations)).precompute(
        n_jobs=n_jobs, save_path=save_path
    )
    return Anatomy.load(save_path)


def _as_row(x):
    return x.iloc[0] if isinstance(x, pd.DataFrame) else x


def extract_shapley_metrics(anatomy_obj: Anatomy, model_key: str) -> Dict[str, Any]:
    groups = {model_key: [model_key]}

    # oShapley-VI (prediction-based): explain predictions
    df_fc = anatomy_obj.explain(
        model_sets=AnatomyModelCombination(groups=groups),
        transformer=AnatomyModelOutputTransformer(transform=lambda y_hat: y_hat),
    )
    block = df_fc.loc[model_key]
    vi = (block.abs().mean(axis=0) if isinstance(block, pd.DataFrame) else block.abs())
    vi = vi.drop("base_contribution", errors="ignore")

    # GPBSV (MSE): explain scalar loss
    df_mse = anatomy_obj.explain(
        model_sets=AnatomyModelCombination(groups=groups),
        transformer=AnatomyModelOutputTransformer(
            transform=lambda y_hat, y: float(np.mean((y - y_hat) ** 2))
        ),
    )
    mse_row = _as_row(df_mse.loc[model_key])
    pbsv_mse = mse_row.drop("base_contribution", errors="ignore")
    base_mse = float(mse_row.get("base_contribution", np.nan))
    total_mse = float(mse_row.sum())

    # GPBSV (RMSE)
    df_rmse = anatomy_obj.explain(
        model_sets=AnatomyModelCombination(groups=groups),
        transformer=AnatomyModelOutputTransformer(
            transform=lambda y_hat, y: float(np.sqrt(np.mean((y - y_hat) ** 2)))
        ),
    )
    rmse_row = _as_row(df_rmse.loc[model_key])
    pbsv_rmse = rmse_row.drop("base_contribution", errors="ignore")
    base_rmse = float(rmse_row.get("base_contribution", np.nan))
    total_rmse = float(rmse_row.sum())

    # MAS
    try:
        mas_res = MAS(vi, pbsv_mse, MAS.LossType.LOWER_IS_BETTER).compute(
            mas_type=MAS.MASType.IMPORTANCE_WEIGHTED,
            hypothesis_test=True,
            h0_alpha=0.5,
        )
    except Exception:
        mas_res = {"mas": np.nan, "mas_p_value": np.nan}

    return dict(
        vi=vi,
        pbsv_mse=pbsv_mse,
        pbsv_rmse=pbsv_rmse,
        base_mse=base_mse,
        total_mse=total_mse,
        base_rmse=base_rmse,
        total_rmse=total_rmse,
        mas=mas_res,
    )


# =============================================================================
# Ground-truth OOS MSE: baseline + refit
# =============================================================================
def baseline_oos_mse(xy: pd.DataFrame, subsets: AnatomySubsets) -> float:
    errs = []
    for period in range(subsets.n_periods):
        train = xy.iloc[subsets.get_train_subset(period)]
        test = xy.iloc[subsets.get_test_subset(period)]
        ybar = float(train["y"].mean())
        yte = float(test["y"].iloc[0])
        errs.append((yte - ybar) ** 2)
    return float(np.mean(errs))


def oos_mse_refit(
    xy: pd.DataFrame,
    subsets: AnatomySubsets,
    model_name: str,
    features: Sequence[str],
    *,
    seed_base: int,
    enet_alpha: float,
) -> float:
    feats = list(features)
    errs = []

    for period in range(subsets.n_periods):
        train = xy.iloc[subsets.get_train_subset(period)]
        test = xy.iloc[subsets.get_test_subset(period)]

        Xtr, ytr = train[feats], train["y"]
        Xte = test[feats]
        yte = float(test["y"].iloc[0])

        est = fit_model(
            model_name,
            Xtr,
            ytr,
            seed_base=seed_base,
            period=period,
            enet_alpha=enet_alpha,
        )
        yhat = float(np.asarray(est.predict(Xte)).ravel()[0])
        errs.append((yte - yhat) ** 2)

    return float(np.mean(errs))


# =============================================================================
# Classical measures
# =============================================================================
def ols_classical_over_periods(xy: pd.DataFrame, subsets: AnatomySubsets) -> pd.DataFrame:
    """
    OLS classical measures across all forecasting windows:
      avg |t-stat|, frac signif, avg standardized |beta|, and APE (= avg |beta|).
    """
    feats = [c for c in xy.columns if c != "y"]
    p = len(feats)

    all_abs_t = []
    all_std_beta = []
    all_abs_beta = []

    for period in range(subsets.n_periods):
        train = xy.iloc[subsets.get_train_subset(period)]
        X = train[feats].values
        y = train["y"].values
        n = X.shape[0]

        Xc = np.column_stack([np.ones(n), X])
        XtX_inv = np.linalg.pinv(Xc.T @ Xc)
        beta = XtX_inv @ (Xc.T @ y)

        resid = y - Xc @ beta
        s2 = float(np.sum(resid ** 2) / max(n - p - 1, 1))
        se = np.sqrt(np.maximum(np.diag(XtX_inv) * s2, 1e-24))

        b = beta[1:]
        abs_t = np.abs(b / np.maximum(se[1:], 1e-12))

        sd_x = np.std(X, axis=0, ddof=1)
        sd_y = max(float(np.std(y, ddof=1)), 1e-12)
        std_b = np.abs(b) * sd_x / sd_y

        all_abs_t.append(abs_t)
        all_std_beta.append(std_b)
        all_abs_beta.append(np.abs(b))

    out = pd.DataFrame(index=feats)
    out["avg_abs_tstat"] = np.mean(np.asarray(all_abs_t), axis=0)
    out["frac_signif_5pct"] = np.mean((np.asarray(all_abs_t) > 1.96), axis=0)
    out["avg_std_beta"] = np.mean(np.asarray(all_std_beta), axis=0)
    out["ape_abs_beta"] = np.mean(np.asarray(all_abs_beta), axis=0)  # APE for linear model
    return out


def enet_stats_over_snapshots(
    xy: pd.DataFrame,
    subsets: AnatomySubsets,
    snaps: Sequence[int],
    *,
    seed_base: int,
    enet_alpha: float,
) -> pd.DataFrame:
    """
    ENet “classical” coefficient-based measures across snapshot windows:
      - APE proxy: avg |beta_j| on original scale
      - standardized coefficient: avg |beta_j| * sd(x_j) / sd(y)
    """
    feats = [c for c in xy.columns if c != "y"]
    abs_coef_list = []
    std_coef_list = []

    for period in snaps:
        train = xy.iloc[subsets.get_train_subset(period)]
        Xtr, ytr = train[feats], train["y"]
        pipe = fit_model(
            "enet", Xtr, ytr,
            seed_base=seed_base, period=period, enet_alpha=enet_alpha
        )

        scaler = pipe.named_steps["scaler"]
        enet = pipe.named_steps["enet"]

        coef_scaled = np.asarray(enet.coef_, dtype=float)
        scale = np.asarray(getattr(scaler, "scale_", np.ones_like(coef_scaled)), dtype=float)
        scale = np.where(np.abs(scale) > 1e-12, scale, 1.0)
        coef_orig = coef_scaled / scale

        abs_coef = np.abs(coef_orig)
        sd_x = Xtr.std(axis=0, ddof=1).values
        sd_y = max(float(ytr.std(ddof=1)), 1e-12)
        std_coef = abs_coef * sd_x / sd_y

        abs_coef_list.append(abs_coef)
        std_coef_list.append(std_coef)

    out = pd.DataFrame(index=feats)
    out["enet_ape_abs_coef"] = np.mean(np.asarray(abs_coef_list), axis=0)
    out["enet_std_coef"] = np.mean(np.asarray(std_coef_list), axis=0)
    return out


def train_perm_importance_over_snapshots(
    xy: pd.DataFrame,
    subsets: AnatomySubsets,
    model_name: str,
    snaps: Sequence[int],
    *,
    seed_base: int,
    enet_alpha: float,
    n_repeats: int,
) -> pd.Series:
    """
    In-sample permutation importance (training window) averaged over snapshots.
    This is intentionally “classical measure that can be fooled.”
    """
    feats = [c for c in xy.columns if c != "y"]
    vals = []

    for period in snaps:
        train = xy.iloc[subsets.get_train_subset(period)]
        Xtr, ytr = train[feats], train["y"]

        est = fit_model(
            model_name, Xtr, ytr,
            seed_base=seed_base, period=period, enet_alpha=enet_alpha
        )

        res = sk_perm_importance(
            est, Xtr, ytr,
            n_repeats=int(n_repeats),
            random_state=int(seed_base + 30_000 + period),
            scoring="neg_mean_squared_error",
        )
        vals.append(pd.Series(res.importances_mean, index=feats))

    return pd.concat(vals, axis=1).mean(axis=1)


def pdp_range_for_estimator(estimator: Any, X: pd.DataFrame, grid_size: int) -> pd.Series:
    """
    PDP range: sweep each feature from its 5th to 95th percentile,
    hold others fixed at mean, compute max(pred) - min(pred).
    """
    cols = list(X.columns)
    x_bar = X.mean(axis=0).values
    out: Dict[str, float] = {}

    for j, col in enumerate(cols):
        q05, q95 = np.quantile(X[col].values, [0.05, 0.95])
        grid = np.linspace(q05, q95, int(grid_size))
        Xg = np.tile(x_bar, (len(grid), 1))
        Xg[:, j] = grid
        pred = estimator.predict(pd.DataFrame(Xg, columns=cols))
        out[col] = float(np.max(pred) - np.min(pred))

    return pd.Series(out)


def pdp_range_over_snapshots(
    xy: pd.DataFrame,
    subsets: AnatomySubsets,
    model_name: str,
    snaps: Sequence[int],
    *,
    seed_base: int,
    enet_alpha: float,
    grid_size: int,
) -> pd.Series:
    feats = [c for c in xy.columns if c != "y"]
    vals = []

    for period in snaps:
        train = xy.iloc[subsets.get_train_subset(period)]
        Xtr, ytr = train[feats], train["y"]

        est = fit_model(
            model_name, Xtr, ytr,
            seed_base=seed_base, period=period, enet_alpha=enet_alpha
        )
        vals.append(pdp_range_for_estimator(est, Xtr, grid_size=grid_size))

    return pd.concat(vals, axis=1).mean(axis=1)


def rf_average_partial_effect(
    rf: RandomForestRegressor,
    X: pd.DataFrame,
    *,
    delta_scale: float,
    max_obs: int,
    random_state: int,
) -> pd.Series:
    """
    Finite-difference APE proxy for RF:
      mean | (f(x+delta e_j) - f(x-delta e_j)) / (2 delta) |
    """
    rng = np.random.default_rng(int(random_state))
    Xs = X.copy()
    if len(Xs) > int(max_obs):
        idx = rng.choice(len(Xs), size=int(max_obs), replace=False)
        Xs = Xs.iloc[idx]

    Xv = Xs.values
    cols = list(Xs.columns)
    out: Dict[str, float] = {}

    for j, col in enumerate(cols):
        sd = float(np.std(Xv[:, j], ddof=1))
        delta = float(delta_scale) * (sd if sd > 1e-12 else 1.0)

        Xp = Xv.copy()
        Xm = Xv.copy()
        Xp[:, j] += delta
        Xm[:, j] -= delta

        dp = rf.predict(Xp)
        dm = rf.predict(Xm)
        deriv = (dp - dm) / (2.0 * delta)
        out[col] = float(np.mean(np.abs(deriv)))

    return pd.Series(out)


def rf_oob_permutation_importance(
    rf: RandomForestRegressor,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_repeats: int,
) -> pd.Series:
    """
    Breiman-style OOB permutation importance:
    mean increase in OOB MSE when permuting feature j.

    Implementation note:
    sklearn’s bootstrap sampling is generated via a RNG;
    we reconstruct bootstrap indices using tree.random_state
    and rs.randint(...). This is the standard practical reconstruction.
    """
    Xv = X.values
    yv = y.values
    n, p = Xv.shape
    cols = list(X.columns)

    n_boot = n  # default for RF bootstrap

    inc = np.zeros(p, dtype=float)
    weight = 0.0

    for tree in rf.estimators_:
        rs = np.random.RandomState(tree.random_state)
        boot_idx = rs.randint(0, n, n_boot)

        oob_mask = np.ones(n, dtype=bool)
        oob_mask[boot_idx] = False
        oob_idx = np.where(oob_mask)[0]
        if oob_idx.size < 5:
            continue

        X_oob = Xv[oob_idx, :]
        y_oob = yv[oob_idx]

        pred0 = tree.predict(X_oob)
        mse0 = np.mean((y_oob - pred0) ** 2)

        for j in range(p):
            mse_rep = 0.0
            for _ in range(int(n_repeats)):
                Xp = X_oob.copy()
                perm = rs.permutation(Xp.shape[0])
                Xp[:, j] = Xp[perm, j]
                predp = tree.predict(Xp)
                mse_rep += np.mean((y_oob - predp) ** 2)
            mse_rep /= float(n_repeats)
            inc[j] += (mse_rep - mse0) * oob_idx.size

        weight += oob_idx.size

    if weight <= 0:
        return pd.Series(np.nan, index=cols)

    inc /= weight
    return pd.Series(inc, index=cols)


def rf_oob_vi_and_ape_over_snapshots(
    xy: pd.DataFrame,
    subsets: AnatomySubsets,
    snaps: Sequence[int],
    *,
    seed_base: int,
    enet_alpha: float,
) -> Tuple[pd.Series, pd.Series]:
    feats = [c for c in xy.columns if c != "y"]
    vi_list = []
    ape_list = []

    for period in snaps:
        train = xy.iloc[subsets.get_train_subset(period)]
        Xtr, ytr = train[feats], train["y"]

        rf = fit_model(
            "rf", Xtr, ytr,
            seed_base=seed_base, period=period, enet_alpha=enet_alpha
        )

        vi_list.append(rf_oob_permutation_importance(rf, Xtr, ytr, n_repeats=RF_OOB_PERM_REPEATS))
        ape_list.append(
            rf_average_partial_effect(
                rf,
                Xtr,
                delta_scale=RF_APE_DELTA_SCALE,
                max_obs=RF_APE_MAX_OBS,
                random_state=seed_base + 40_000 + period,
            )
        )

    vi = pd.concat(vi_list, axis=1).mean(axis=1)
    ape = pd.concat(ape_list, axis=1).mean(axis=1)
    return vi, ape


# =============================================================================
# Success evaluation logic
# =============================================================================
def topk_flag(ranks: Dict[str, int], k: int) -> bool:
    return all(v <= k for v in ranks.values())


def rank1_flag(ranks: Dict[str, int]) -> bool:
    return all(v == 1 for v in ranks.values())


# =============================================================================
# Runner
# =============================================================================
def run_scenario(sc: Scenario) -> None:
    print("\n" + "=" * 100)
    print(f"SCENARIO: {sc.name}")
    print("=" * 100)
    print(f"DGP kind: {sc.dgp_kind} | n_total={sc.n_total} | initial_window={sc.initial_window} | estimation={sc.estimation_type}")
    print(f"Params: {sc.params}")
    print(f"Seeds: {SEEDS[0]}..{SEEDS[-1]} (n={len(SEEDS)})")
    print(f"Models: {MODEL_NAMES}")
    print(f"Shapley iterations: {N_ITERATIONS_SHAPLEY}")
    print(f"Snapshots per seed: {N_SNAPSHOTS}")
    print(f"EPS={EPS} | Smoking-gun improvement threshold={IMPROVE_PCT_MIN:.1f}%")
    print()

    out_dir = ensure_dir(os.path.join(OUTPUT_ROOT, sc.name))
    rows: List[Dict[str, Any]] = []

    for i, seed in enumerate(SEEDS, start=1):
        # --- Generate data
        if sc.dgp_kind == "A":
            xy = dgp_structural_break(
                n_total=sc.n_total,
                break_at=int(sc.params["break_at"]),
                initial_window=int(sc.initial_window),
                rho_x=float(sc.params["rho_x"]),
                rho_noise=float(sc.params["rho_noise"]),
                beta1=float(sc.params["beta1"]),
                beta2_pre=float(sc.params["beta2_pre"]),
                beta2_post=float(sc.params["beta2_post"]),
                sigma_eps=float(sc.params["sigma_eps"]),
                seed=int(seed),
            )
        elif sc.dgp_kind == "B":
            xy = dgp_persistent_noise(
                n_total=sc.n_total,
                rho_signal=float(sc.params["rho_signal"]),
                rho_trap=float(sc.params["rho_trap"]),
                rho_noise=float(sc.params["rho_noise"]),
                beta1=float(sc.params["beta1"]),
                sigma_eps=float(sc.params["sigma_eps"]),
                seed=int(seed),
            )
        elif sc.dgp_kind == "C":
            xy = dgp_proxy_breakdown(
                n_total=sc.n_total,
                break_at=int(sc.params["break_at"]),
                rho_signal=float(sc.params["rho_signal"]),
                rho_proxy_noise=float(sc.params["rho_proxy_noise"]),
                rho_noise=float(sc.params["rho_noise"]),
                beta_signal=float(sc.params["beta_signal"]),
                loading_pre=float(sc.params["loading_pre"]),
                loading_post=float(sc.params["loading_post"]),
                sigma_x1_me=float(sc.params["sigma_x1_me"]),
                sigma_x2_me=float(sc.params["sigma_x2_me"]),
                sigma_eps=float(sc.params["sigma_eps"]),
                seed=int(seed),
            )
        else:
            raise ValueError("Unknown dgp_kind")

        # --- Forecast subsets
        est_type = (
            AnatomySubsets.EstimationType.EXPANDING
            if sc.estimation_type == "expanding"
            else AnatomySubsets.EstimationType.ROLLING
        )
        subsets = AnatomySubsets.generate(
            index=xy.index,
            initial_window=int(sc.initial_window),
            estimation_type=est_type,
            periods=1,
            gap=0,
        )

        # --- ENet alpha once per seed (initial window)
        init_train = xy.iloc[subsets.get_train_subset(0)]
        enet_alpha = select_enet_alpha_once(init_train[FEATURES], init_train["y"], seed=int(seed))

        # --- Anatomy
        bin_path = os.path.join(out_dir, f"anatomy_seed{seed}.bin")
        anatomy_obj = run_anatomy(
            xy,
            subsets,
            MODEL_NAMES,
            n_iterations=N_ITERATIONS_SHAPLEY,
            save_path=bin_path,
            seed_base=int(seed),
            enet_alpha=float(enet_alpha),
        )

        # --- Baseline
        mse_base = baseline_oos_mse(xy, subsets)

        # --- Snapshot periods for expensive measures
        snaps = snapshot_periods(subsets.n_periods, n_snapshots=N_SNAPSHOTS)

        # --- Classical OLS measures (cheap: all windows)
        classical_ols = ols_classical_over_periods(xy, subsets)

        # --- ENet coefficient-based measures (snapshots)
        enet_stats = enet_stats_over_snapshots(
            xy, subsets, snaps, seed_base=int(seed), enet_alpha=float(enet_alpha)
        )

        # --- RF-specific measures (snapshots)
        rf_oob_vi, rf_ape = rf_oob_vi_and_ape_over_snapshots(
            xy, subsets, snaps, seed_base=int(seed), enet_alpha=float(enet_alpha)
        )

        status_bits: List[str] = []

        for model_name in MODEL_NAMES:
            # Shapley metrics
            met = extract_shapley_metrics(anatomy_obj, model_name)
            vi = met["vi"].reindex(FEATURES)
            pbsv_mse = met["pbsv_mse"].reindex(FEATURES)
            pbsv_rmse = met["pbsv_rmse"].reindex(FEATURES)

            x2_vi_rank = int(rank_desc(vi).loc["x2"])
            x2_pbsv = float(pbsv_mse.loc["x2"])

            # Ground-truth refit MSE: full vs drop-x2
            mse_full = oos_mse_refit(
                xy, subsets, model_name, FEATURES, seed_base=int(seed), enet_alpha=float(enet_alpha)
            )
            mse_drop = oos_mse_refit(
                xy, subsets, model_name, ["x1", "x3", "x4"], seed_base=int(seed), enet_alpha=float(enet_alpha)
            )
            improve_pct = 100.0 * (mse_full - mse_drop) / max(mse_full, 1e-12)

            # In-sample permutation importance (snapshots)
            perm_imp = train_perm_importance_over_snapshots(
                xy, subsets, model_name, snaps,
                seed_base=int(seed),
                enet_alpha=float(enet_alpha),
                n_repeats=PERM_REPEATS_TRAIN,
            ).reindex(FEATURES)

            # PDP range (snapshots)
            pdp_rng = pdp_range_over_snapshots(
                xy, subsets, model_name, snaps,
                seed_base=int(seed),
                enet_alpha=float(enet_alpha),
                grid_size=PDP_GRID_SIZE,
            ).reindex(FEATURES)

            # Build model-specific “classical” rank set for x2
            if model_name == "ols":
                classical_measures = {
                    "abs_t": classical_ols["avg_abs_tstat"].reindex(FEATURES),
                    "std_beta": classical_ols["avg_std_beta"].reindex(FEATURES),
                    "ape": classical_ols["ape_abs_beta"].reindex(FEATURES),
                    "perm": perm_imp,
                    "pdp": pdp_rng,
                }
            elif model_name == "enet":
                classical_measures = {
                    "std_coef": enet_stats["enet_std_coef"].reindex(FEATURES),
                    "ape": enet_stats["enet_ape_abs_coef"].reindex(FEATURES),
                    "perm": perm_imp,
                    "pdp": pdp_rng,
                }
            elif model_name == "rf":
                classical_measures = {
                    "oob_vi": rf_oob_vi.reindex(FEATURES),
                    "rf_ape": rf_ape.reindex(FEATURES),
                    "perm": perm_imp,
                    "pdp": pdp_rng,
                }
            else:
                raise ValueError("Unknown model_name")

            x2_classical_ranks = {k: int(rank_desc(v).loc["x2"]) for k, v in classical_measures.items()}
            x2_oshapley_rank = int(rank_desc(vi).loc["x2"])

            x2_harmful = x2_pbsv > EPS
            drop_improves = mse_drop < mse_full

            x2_top2_all_classical = topk_flag(x2_classical_ranks, k=2)
            x2_rank1_all_classical = rank1_flag(x2_classical_ranks)

            x2_top2_oshapley = (x2_oshapley_rank <= 2)
            x2_rank1_oshapley = (x2_oshapley_rank == 1)

            # Primary success (the one you can safely summarize in the paper)
            primary_success = bool(x2_harmful and drop_improves and x2_top2_all_classical and x2_top2_oshapley)

            # Smoking-gun variants
            smoking_gun_top2 = bool(
                x2_harmful
                and drop_improves
                and (improve_pct >= IMPROVE_PCT_MIN)
                and x2_top2_all_classical
                and x2_top2_oshapley
            )
            strict_rank1 = bool(
                x2_harmful
                and drop_improves
                and (improve_pct >= IMPROVE_PCT_MIN)
                and x2_rank1_all_classical
                and x2_rank1_oshapley
            )

            model_beats_baseline = mse_full < mse_base

            # Build and save per-seed comparison table (appendix-ready)
            tbl = pd.DataFrame(index=FEATURES)
            tbl["true_role"] = pd.Series(TRUE_ROLE)
            tbl["oShapley_VI"] = vi
            tbl["VI_rank"] = rank_desc(vi)
            tbl["GPBSV_MSE"] = pbsv_mse
            tbl["GPBSV_RMSE"] = pbsv_rmse
            tbl["interpretation"] = tbl["GPBSV_MSE"].apply(lambda z: interpret_pbsv(float(z)))

            if model_name == "ols":
                tbl["avg_abs_tstat"] = classical_ols["avg_abs_tstat"].reindex(FEATURES)
                tbl["frac_signif_5pct"] = classical_ols["frac_signif_5pct"].reindex(FEATURES)
                tbl["avg_std_beta"] = classical_ols["avg_std_beta"].reindex(FEATURES)
                tbl["ape_abs_beta"] = classical_ols["ape_abs_beta"].reindex(FEATURES)

            if model_name == "enet":
                tbl["enet_std_coef"] = enet_stats["enet_std_coef"].reindex(FEATURES)
                tbl["enet_ape_abs_coef"] = enet_stats["enet_ape_abs_coef"].reindex(FEATURES)

            if model_name == "rf":
                tbl["rf_oob_vi"] = rf_oob_vi.reindex(FEATURES)
                tbl["rf_ape"] = rf_ape.reindex(FEATURES)

            tbl["perm_import_train"] = perm_imp
            tbl["pdp_range"] = pdp_rng

            tbl_path = os.path.join(out_dir, f"table_{model_name}_seed{seed}.csv")
            tbl.to_csv(tbl_path)

            # Store summary row
            row: Dict[str, Any] = dict(
                scenario=sc.name,
                seed=int(seed),
                model=model_name,
                n_periods=int(subsets.n_periods),
                enet_alpha=float(enet_alpha),
                mse_baseline=float(mse_base),
                mse_full=float(mse_full),
                mse_drop_x2=float(mse_drop),
                improve_pct=float(improve_pct),
                x2_pbsv_mse=float(x2_pbsv),
                x2_oshapley_rank=int(x2_oshapley_rank),
                primary_success=primary_success,
                smoking_gun_top2=smoking_gun_top2,
                strict_rank1=strict_rank1,
                model_beats_baseline=bool(model_beats_baseline),
                drop_x2_improves=bool(drop_improves),
                mas=float(met["mas"].get("mas", np.nan)),
                mas_p=float(met["mas"].get("mas_p_value", np.nan)),
            )
            for k, v in x2_classical_ranks.items():
                row[f"rank_x2_{k}"] = int(v)

            rows.append(row)

            # Console status tag
            tag = "S1" if strict_rank1 else ("SG" if smoking_gun_top2 else ("OK" if primary_success else "XX"))
        status_bits.append(f"{model_name}:{tag}(VI={x2_vi_rank},GPBSV={x2_pbsv:+.3f},improve={improve_pct:+.1f}%)")

        print(f"[{i:02d}/{len(SEEDS)}] seed={seed} | " + " | ".join(status_bits))

        if CLEANUP_ANATOMY_BIN:
            try:
                os.remove(bin_path)
            except OSError:
                pass

    # --- Scenario summary
    df = pd.DataFrame(rows)
    summary_path = os.path.join(out_dir, "grand_summary.csv")
    df.to_csv(summary_path, index=False)

    print("\n" + "-" * 100)
    print("SUCCESS RATES")
    print("-" * 100)

    for m in MODEL_NAMES:
        sub = df[df["model"] == m].copy()
        n = len(sub)
        if n == 0:
            continue

        def _rate(col: str) -> str:
            k = int(sub[col].sum())
            return f"{k}/{n} = {k/n:.0%}"

        print(f"\n{m.upper()} (n={n})")
        print(f"  primary_success:     {_rate('primary_success')}")
        print(f"  smoking_gun_top2:    {_rate('smoking_gun_top2')}")
        print(f"  strict_rank1:        {_rate('strict_rank1')}")
        print(f"  full beats baseline: {_rate('model_beats_baseline')}")
        print(f"  drop-x2 improves:    {_rate('drop_x2_improves')}")
        print(f"  avg GPBSV(x2):       {sub['x2_pbsv_mse'].mean():+.4f}")
        print(f"  avg improve% (drop x2): {sub['improve_pct'].mean():+.1f}%")
        print(f"  avg oShapley rank:   {sub['x2_oshapley_rank'].mean():.2f}")
        if m == "enet":
            print(f"  ENet alpha (mean):   {sub['enet_alpha'].mean():.4g}")
            print(f"  ENet alpha (min/max): {sub['enet_alpha'].min():.4g} / {sub['enet_alpha'].max():.4g}")

        rank_cols = sorted([c for c in sub.columns if c.startswith("rank_x2_")])
        if rank_cols:
            print("  x2 rank-1 and top-2 rates by measure:")
            for c in rank_cols:
                r1 = float((sub[c] == 1).mean())
                r2 = float((sub[c] <= 2).mean())
                print(f"    {c.replace('rank_x2_', ''):>10s}: rank1={r1:.0%} | top2={r2:.0%}")

    # --- Representative exhibits
    print("\n" + "-" * 100)
    print("REPRESENTATIVE EXHIBITS (per model)")
    print("-" * 100)

    pd.options.display.width = 220
    pd.options.display.float_format = "{:.4f}".format

    for m in MODEL_NAMES:
        sub = df[df["model"] == m].copy()

        # Selection rule:
        # 1) Prefer smoking_gun_top2 seeds that also beat baseline
        # 2) Else smoking_gun_top2
        # 3) Else primary_success
        pool = sub[(sub["smoking_gun_top2"]) & (sub["model_beats_baseline"])]
        if len(pool) == 0:
            pool = sub[sub["smoking_gun_top2"]]
        if len(pool) == 0:
            pool = sub[sub["primary_success"]]
        if len(pool) == 0:
            print(f"\n{m.upper()}: No successful seeds.")
            continue

        best = pool.sort_values(["improve_pct"], ascending=False).iloc[0]
        seed = int(best["seed"])

        tbl_path = os.path.join(out_dir, f"table_{m}_seed{seed}.csv")
        tbl = pd.read_csv(tbl_path, index_col=0)

        print(f"\n--- {m.upper()} EXHIBIT (seed={seed}) ---")
        print(f"MSE baseline={best['mse_baseline']:.4f} | full={best['mse_full']:.4f} | drop-x2={best['mse_drop_x2']:.4f}")
        print(f"improve% (drop x2)={best['improve_pct']:.1f}% | x2 GPBSV={best['x2_pbsv_mse']:+.4f} | x2 oShapley rank={best['x2_oshapley_rank']}")
        print(f"MAS={best['mas']:.4f} (p={best['mas_p']:.4f})")

        cols_pref = [
            "true_role",
            "avg_abs_tstat", "frac_signif_5pct", "avg_std_beta", "ape_abs_beta",
            "enet_std_coef", "enet_ape_abs_coef",
            "rf_oob_vi", "rf_ape",
            "perm_import_train", "pdp_range",
            "oShapley_VI", "VI_rank",
            "GPBSV_MSE", "interpretation",
        ]
        cols_show = [c for c in cols_pref if c in tbl.columns]
        # Sort by VI rank for readability
        tbl2 = tbl[cols_show].copy()
        if "VI_rank" in tbl2.columns:
            tbl2 = tbl2.sort_values("VI_rank")
        print(tbl2.to_string())

    print("\nSaved outputs to:", out_dir)
    print("Grand summary:", summary_path)
    print("Done.")


def main():
    global OUTPUT_ROOT, N_SEEDS, SEEDS, N_ITERATIONS_SHAPLEY

    parser = argparse.ArgumentParser(description="Run the Simulation 1 trap-DGP revision study.")
    parser.add_argument("--output-root", type=str, default=OUTPUT_ROOT)
    parser.add_argument("--n-seeds", type=int, default=N_SEEDS)
    parser.add_argument("--seed-start", type=int, default=SEEDS[0])
    parser.add_argument("--n-iterations-shapley", type=int, default=N_ITERATIONS_SHAPLEY)
    parser.add_argument(
        "--scenario",
        action="append",
        choices=[sc.name for sc in SCENARIOS],
        help="Limit the run to one or more named scenarios.",
    )
    args = parser.parse_args()

    OUTPUT_ROOT = os.path.abspath(args.output_root)
    N_SEEDS = int(args.n_seeds)
    SEEDS = list(range(int(args.seed_start), int(args.seed_start) + N_SEEDS))
    N_ITERATIONS_SHAPLEY = int(args.n_iterations_shapley)

    ensure_dir(OUTPUT_ROOT)
    selected = set(args.scenario) if args.scenario else None
    scenarios = [sc for sc in SCENARIOS if selected is None or sc.name in selected]
    for sc in scenarios:
        run_scenario(sc)


if __name__ == "__main__":
    main()
