"""
Simulation 2 (Referee R3): Shapley vs Average Partial Effect vs OOB Importance
-----------------------------------------------------------------------------

Goal (simple, table-only output):
- Controlled nonlinear DGP with an interaction term that a tree model can learn.
- Compare (i) TS-Shapley-VI and (ii) GPBSV (RMSE) from the anatomy package
  to (iii) Average Partial Effects (APE) and (iv) OOB permutation importance (RF).

Key design choice (to make the comparison meaningful):
- DGP: y_{t+1} = beta_int * x1_t * x2_t + beta3 * x3_t + eps_{t+1}
  with x1,x2 mean ~0, so the *signed* APE of x1 and x2 is ~0 even though they matter a lot.
  This makes a clean point for the referee: APE is not a variable-importance measure under
  interactions; Shapley-VI and OOB importance still flag x1/x2 as important.

Outputs:
- Creates ./simulation_2/ and saves:
  - xy.csv (the simulated dataset)
  - TS_Shapley_VI.csv
  - GPBSV_RMSE.csv
  - TS_APE_and_OOB.csv
  - TABLE_ols.csv
  - TABLE_rf.csv
  - SUMMARY.csv

Environment:
- Python 3.9 compatible (no "type | None" syntax).
- Tested for the imports and syntax style used in your existing scripts.
"""

import argparse
import os
import json
import warnings

import numpy as np
import pandas as pd

from anatomy import *

from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import check_random_state

# Private sklearn helper (works in sklearn 0.24.*)
from sklearn.ensemble._forest import _generate_sample_indices

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SIMULATION_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
DEFAULT_OUT_DIR = os.path.join(SIMULATION_ROOT, "outputs", "sim2")


# -----------------------------
# 1) DGP: AR(1) predictors + interaction signal
# -----------------------------
def _ar1_stationary(n, rho, rng):
    """
    Stationary AR(1): x_t = rho x_{t-1} + sqrt(1-rho^2) * e_t, e_t ~ N(0,1)
    => Var(x_t)=1 approximately.
    """
    x = np.empty(n)
    e = rng.normal(0.0, 1.0, size=n)
    x[0] = e[0]
    scale = np.sqrt(max(1e-12, 1.0 - rho * rho))
    for t in range(1, n):
        x[t] = rho * x[t - 1] + scale * e[t]
    return x


def generate_interaction_dgp(
    n_total_obs=280,          # total raw time points for x_t (we will use rows t=1..n_total_obs-1)
    p=6,                      # x1..xp predictors
    rho=0.5,                  # AR(1) persistence for all predictors
    beta_int=2.0,             # coefficient on x1*x2
    beta3=0.5,                # coefficient on x3
    sigma_eps=1.0,
    seed=1338,
):
    """
    Build a time-indexed dataframe with predictors at time t and target y_{t+1}.

    True DGP:
      y_{t+1} = beta_int * x1_t * x2_t + beta3 * x3_t + eps_{t+1}
      x4..xp irrelevant
    """
    assert p >= 3, "Need at least 3 predictors for this DGP."

    rng = np.random.default_rng(seed)

    # predictors X_t for t=0..n_total_obs-1
    X = np.zeros((n_total_obs, p))
    for j in range(p):
        X[:, j] = _ar1_stationary(n_total_obs, rho=rho, rng=rng)

    eps = rng.normal(0.0, sigma_eps, size=n_total_obs)

    # y_{t+1} aligned with predictors at t -> rows use X[t] and y_next[t]=y_{t+1}
    y_next = beta_int * (X[:-1, 0] * X[:-1, 1]) + beta3 * X[:-1, 2] + eps[1:]

    cols = [f"x{j+1}" for j in range(p)]
    df = pd.DataFrame(X[:-1, :], columns=cols)
    df["y"] = y_next

    # time index (daily just for unique dates)
    df.index = pd.date_range("2000-01-01", periods=len(df)).map(lambda x: x.date())

    return df


# -----------------------------
# 2) Anatomy model wrappers
# -----------------------------
def train_ols_anatomy(x_train_df, y_train_s):
    ols = LinearRegression().fit(x_train_df, y_train_s)

    def pred_fn(xs):
        xs_df = pd.DataFrame(xs, columns=x_train_df.columns)
        return np.asarray(ols.predict(xs_df)).flatten()

    return AnatomyModel(pred_fn)


def train_rf_anatomy(x_train_df, y_train_s, random_state=1338):
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=5,
        min_samples_split=10,
        max_features="auto",
        bootstrap=True,
        oob_score=True,
        n_jobs=1,              # keep deterministic / avoid nested parallelism
        random_state=random_state,
    ).fit(x_train_df, y_train_s)

    def pred_fn(xs):
        xs_df = pd.DataFrame(xs, columns=x_train_df.columns)
        return np.asarray(rf.predict(xs_df)).flatten()

    return AnatomyModel(pred_fn)


def estimate_anatomy(
    xy,
    subsets,
    n_iterations=50,               # Shapley permutation draws (M)
    save_path="anatomy_sim2.bin",
    n_jobs=1,
    base_seed_for_models=1338,
):
    """
    Precompute Anatomy objects for two models: OLS and RF.
    """

    def mapper(key):
        train = xy.iloc[subsets.get_train_subset(key.period)]
        test = xy.iloc[subsets.get_test_subset(key.period)]

        x_train = train.drop("y", axis=1)
        y_train = train["y"]

        if key.model_name == "ols":
            model = train_ols_anatomy(x_train, y_train)
        elif key.model_name == "rf":
            # vary RF seed by period so forests are not identical across windows
            model = train_rf_anatomy(x_train, y_train, random_state=base_seed_for_models + key.period)
        else:
            raise ValueError("Unknown model_name: %s" % key.model_name)

        return AnatomyModelProvider.PeriodValue(train, test, model)

    provider = AnatomyModelProvider(
        n_periods=subsets.n_periods,
        n_features=xy.shape[1] - 1,
        model_names=["ols", "rf"],
        y_name="y",
        provider_fn=mapper,
    )

    Anatomy(provider=provider, n_iterations=n_iterations).precompute(
        n_jobs=n_jobs,
        save_path=save_path,
    )


# -----------------------------
# 3) Extract TS-Shapley-VI and GPBSV-RMSE from Anatomy
# -----------------------------
def _extract_ts_shapley_vi(df_forecasts, model_name):
    """
    df_forecasts is typically indexed by (model_name, date), columns are base_contribution + x1..xp.
    TS-Shapley-VI is computed as average |contribution| across dates.
    """
    sel = df_forecasts.loc[model_name]
    if isinstance(sel, pd.Series):
        vi = sel.abs()
    else:
        vi = sel.abs().mean(axis=0)
    vi = vi.drop("base_contribution", errors="ignore")
    return vi


def _extract_gpbsv_rmse(df_pbsv_rmse, model_name):
    """
    df_pbsv_rmse is typically indexed by (model_name, "start -> end"), one row per model_name.
    Returns:
      pbsv (Series over predictors),
      rmse_total (float),
      full_row (Series including base_contribution).
    """
    sel = df_pbsv_rmse.loc[model_name]
    if isinstance(sel, pd.Series):
        row = sel
    else:
        row = sel.iloc[0]

    rmse_total = float(row.sum())
    pbsv = row.drop("base_contribution", errors="ignore")
    return pbsv, rmse_total, row


def compute_ts_shapley_vi_and_gpbsv(anatomy_obj):
    groups = {"ols": ["ols"], "rf": ["rf"]}

    def transform_forecasts(y_hat):
        return y_hat

    df_forecasts = anatomy_obj.explain(
        model_sets=AnatomyModelCombination(groups=groups),
        transformer=AnatomyModelOutputTransformer(transform=transform_forecasts),
    )

    def transform_rmse(y_hat, y):
        return np.sqrt(np.mean((y - y_hat) ** 2))

    df_pbsv_rmse = anatomy_obj.explain(
        model_sets=AnatomyModelCombination(groups=groups),
        transformer=AnatomyModelOutputTransformer(transform=transform_rmse),
    )

    out = {}
    for m in ["ols", "rf"]:
        vi = _extract_ts_shapley_vi(df_forecasts, m)
        pbsv, rmse_total, _row = _extract_gpbsv_rmse(df_pbsv_rmse, m)
        out[m] = {"ts_shapley_vi": vi, "gpbsv_rmse": pbsv, "rmse": rmse_total}

    return out, df_forecasts, df_pbsv_rmse


# -----------------------------
# 4) Average Partial Effects (APE) and OOB permutation importance
# -----------------------------
def local_partial_effects(model_predict_fn, x0_row, delta=0.10):
    """
    Finite-difference local partial effects at a single point x0.
    Returns vector of d f(x)/d x_j approximated by central differences.
    """
    x0 = np.asarray(x0_row, dtype=float).reshape(1, -1)
    p = x0.shape[1]
    pe = np.zeros(p)

    for j in range(p):
        x_plus = x0.copy()
        x_minus = x0.copy()
        x_plus[0, j] += delta
        x_minus[0, j] -= delta

        f_plus = float(model_predict_fn(x_plus)[0])
        f_minus = float(model_predict_fn(x_minus)[0])
        pe[j] = (f_plus - f_minus) / (2.0 * delta)

    return pe


def _get_n_samples_bootstrap(rf, n_samples: int) -> int:
    """
    Number of samples drawn in each bootstrap sample.
    In sklearn, this is n_samples if max_samples is None.
    """
    max_samples = getattr(rf, "max_samples", None)

    if max_samples is None:
        return int(n_samples)

    # sklearn supports int or float for max_samples
    if isinstance(max_samples, float):
        # mimic sklearn behavior: fraction of n_samples
        return int(np.floor(max_samples * n_samples))
    return int(max_samples)


def _oob_indices_for_tree(tree, n_samples: int, n_samples_bootstrap: int) -> np.ndarray:
    """
    Reproduce which samples were bootstrapped for this tree using its random_state,
    then return the indices that were NOT sampled (OOB indices).
    """
    rng = check_random_state(tree.random_state)

    # This matches sklearn's bootstrap sampling logic: draw with replacement
    sample_indices = rng.randint(0, n_samples, size=n_samples_bootstrap)

    sample_counts = np.bincount(sample_indices, minlength=n_samples)
    oob_indices = np.flatnonzero(sample_counts == 0)

    return oob_indices


def oob_permutation_importance_mse(
    rf,
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_seed: int = 1338,
    n_repeats: int = 1,
) -> np.ndarray:
    """
    Breiman-style OOB permutation importance for RandomForestRegressor.

    Returns:
      importances: shape (n_features,), where larger positive means more important
                   (bigger increase in OOB MSE after permuting that feature).
    """
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train).ravel()

    n_samples, n_features = X_train.shape

    if not getattr(rf, "bootstrap", True):
        raise ValueError("OOB importance requires bootstrap=True in RandomForestRegressor.")

    n_boot = _get_n_samples_bootstrap(rf, n_samples)
    perm_rng = np.random.RandomState(random_seed)

    importances = np.zeros(n_features, dtype=float)
    n_trees_used = 0

    for tree in rf.estimators_:
        oob_idx = _oob_indices_for_tree(tree, n_samples=n_samples, n_samples_bootstrap=n_boot)
        if oob_idx.size == 0:
            continue

        n_trees_used += 1

        X_oob = X_train[oob_idx, :]
        y_oob = y_train[oob_idx]

        # baseline OOB MSE for this tree
        pred_base = tree.predict(X_oob)
        mse_base = np.mean((y_oob - pred_base) ** 2)

        # permute each feature inside the OOB sample
        for j in range(n_features):
            mse_perm = 0.0
            for _ in range(n_repeats):
                Xp = X_oob.copy()
                perm = perm_rng.permutation(Xp.shape[0])
                Xp[:, j] = Xp[perm, j]
                pred_perm = tree.predict(Xp)
                mse_perm += np.mean((y_oob - pred_perm) ** 2)

            mse_perm /= float(n_repeats)
            importances[j] += (mse_perm - mse_base)

    if n_trees_used == 0:
        raise RuntimeError(
            "No trees had non-empty OOB samples. Ensure bootstrap=True and n_samples is not tiny."
        )

    importances /= float(n_trees_used)
    return importances


def compute_ts_ape_and_oob(
    xy,
    subsets,
    delta=0.10,
    rf_n_estimators=200,
    rf_max_depth=6,
    rf_min_samples_leaf=5,
    rf_min_samples_split=10,
    rf_max_features="auto",
    seed=1338,
):
    """
    Time-series averages across forecast origins (periods):

    - APE (signed): mean of local partial effects at each out-of-sample x_{t} point.
    - absAPE: mean of absolute local partial effects.
    - OOB importance (RF only): mean OOB permutation MSE increase across periods.

    Returns a dict with keys 'ols' and 'rf' with Series by feature name.
    """
    feature_names = [c for c in xy.columns if c != "y"]
    p = len(feature_names)

    # Accumulators
    pe_sum = {"ols": np.zeros(p), "rf": np.zeros(p)}
    pe_abs_sum = {"ols": np.zeros(p), "rf": np.zeros(p)}
    oob_sum = np.zeros(p)
    oob_count = 0

    for period in tqdm(range(subsets.n_periods), desc="TS-APE/OOB loop"):
        train = xy.iloc[subsets.get_train_subset(period)]
        test = xy.iloc[subsets.get_test_subset(period)]

        x_train = train[feature_names]
        y_train = train["y"]
        x_test = test[feature_names].values  # usually shape (1,p)

        # ---- OLS ----
        ols = LinearRegression().fit(x_train, y_train)

        def ols_pred_fn(X):
            X_df = pd.DataFrame(X, columns=feature_names)
            return np.asarray(ols.predict(X_df)).flatten()

        pe_ols = local_partial_effects(ols_pred_fn, x_test[0, :], delta=delta)
        pe_sum["ols"] += pe_ols
        pe_abs_sum["ols"] += np.abs(pe_ols)

        # ---- RF ----
        rf = RandomForestRegressor(
            n_estimators=rf_n_estimators,
            max_depth=rf_max_depth,
            min_samples_leaf=rf_min_samples_leaf,
            min_samples_split=rf_min_samples_split,
            max_features=rf_max_features,
            bootstrap=True,
            oob_score=True,
            n_jobs=1,
            random_state=seed + period,
        ).fit(x_train, y_train)

        def rf_pred_fn(X):
            X_df = pd.DataFrame(X, columns=feature_names)
            return np.asarray(rf.predict(X_df)).flatten()

        pe_rf = local_partial_effects(rf_pred_fn, x_test[0, :], delta=delta)
        pe_sum["rf"] += pe_rf
        pe_abs_sum["rf"] += np.abs(pe_rf)

        # OOB importance for RF
        oob_imp = oob_permutation_importance_mse(rf, x_train.values, y_train.values, random_seed=seed + period)
        oob_sum += oob_imp
        oob_count += 1

    # averages across periods
    nP = float(subsets.n_periods)

    out = {}
    out["ols"] = {
        "APE": pd.Series(pe_sum["ols"] / nP, index=feature_names),
        "absAPE": pd.Series(pe_abs_sum["ols"] / nP, index=feature_names),
        "OOB_MSE_increase": pd.Series([np.nan] * p, index=feature_names),
    }
    out["rf"] = {
        "APE": pd.Series(pe_sum["rf"] / nP, index=feature_names),
        "absAPE": pd.Series(pe_abs_sum["rf"] / nP, index=feature_names),
        "OOB_MSE_increase": pd.Series(oob_sum / float(max(1, oob_count)), index=feature_names),
    }

    return out


# -----------------------------
# 5) Build tables (CSV + print)
# -----------------------------
def build_model_table(model_name, anatomy_metrics, ts_ape_oob, true_effect_map):
    """
    anatomy_metrics: dict with keys ts_shapley_vi, gpbsv_rmse, rmse
    ts_ape_oob: dict with keys APE, absAPE, OOB_MSE_increase
    """
    vi = anatomy_metrics["ts_shapley_vi"].copy()
    pbsv = anatomy_metrics["gpbsv_rmse"].copy()

    # align indices
    features = sorted(list(set(vi.index).intersection(set(pbsv.index))))
    vi = vi.loc[features]
    pbsv = pbsv.loc[features]

    ape = ts_ape_oob["APE"].loc[features]
    absape = ts_ape_oob["absAPE"].loc[features]
    oob = ts_ape_oob["OOB_MSE_increase"].loc[features]

    df = pd.DataFrame(
        {
            "true_effect": [true_effect_map.get(f, "") for f in features],
            "GPBSV_RMSE": pbsv.values,
            "TS_Shapley_VI": vi.values,
            "APE": ape.values,
            "absAPE": absape.values,
            "OOB_MSE_increase": oob.values,
        },
        index=features,
    )

    # ranks (paper style: sort by GPBSV for "helping performance")
    # more negative GPBSV is better -> rank ascending
    df["rank_GPBSV_best"] = df["GPBSV_RMSE"].rank(ascending=True, method="min").astype(int)
    df["rank_TS_Shapley_VI"] = df["TS_Shapley_VI"].rank(ascending=False, method="min").astype(int)
    df["rank_absAPE"] = df["absAPE"].rank(ascending=False, method="min").astype(int)

    if model_name == "rf":
        df["rank_OOB"] = df["OOB_MSE_increase"].rank(ascending=False, method="min").astype(int)
    else:
        df["rank_OOB"] = np.nan

    # useful normalizations (optional, but handy for comparing magnitudes)
    rmse_total = float(anatomy_metrics["rmse"])
    df["GPBSV_share_of_RMSE"] = df["GPBSV_RMSE"] / (rmse_total + 1e-12)

    vi_sum = float(df["TS_Shapley_VI"].sum())
    df["TS_Shapley_VI_share"] = df["TS_Shapley_VI"] / (vi_sum + 1e-12)

    # sort like the paper's Figure 1: by GPBSV (most negative on top)
    df = df.sort_values("GPBSV_RMSE", ascending=True)

    return df


def spearman_corr(a, b):
    """
    Spearman correlation using pandas.
    """
    a = pd.Series(a).copy()
    b = pd.Series(b).copy()
    df = pd.concat([a, b], axis=1).dropna()
    if df.shape[0] < 3:
        return np.nan
    return float(df.iloc[:, 0].corr(df.iloc[:, 1], method="spearman"))


def main():
    parser = argparse.ArgumentParser(description="Run the Simulation 2 APE/OOB comparison study.")
    parser.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR)
    parser.add_argument("--M", type=int, default=30)
    parser.add_argument("--n-total-obs", type=int, default=280)
    parser.add_argument("--initial-window", type=int, default=200)
    parser.add_argument("--rf-n-estimators", type=int, default=200)
    parser.add_argument("--anatomy-jobs", type=int, default=1)
    parser.add_argument("--smoke", action="store_true", help="Use a smaller, faster configuration for verification.")
    args = parser.parse_args()

    # -------------------------
    # USER SETTINGS
    # -------------------------
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    seed = 20260217

    # DGP params
    n_total_obs = int(args.n_total_obs)
    p = 6
    rho = 0.5
    beta_int = 2.0
    beta3 = 0.5
    sigma_eps = 1.0

    # forecasting loop
    initial_window = int(args.initial_window)
    estimation_type = AnatomySubsets.EstimationType.ROLLING
    periods = 1
    gap = 0

    # anatomy
    M = int(args.M)
    anatomy_jobs = int(args.anatomy_jobs)

    # APE finite diff
    delta = 0.10

    # RF params for APE/OOB loop (separate from anatomy fitting; keep consistent)
    rf_n_estimators = int(args.rf_n_estimators)
    rf_max_depth = 6
    rf_min_samples_leaf = 5
    rf_min_samples_split = 10
    rf_max_features = "auto"

    if args.smoke:
        n_total_obs = 120
        initial_window = 60
        M = 5
        anatomy_jobs = 1
        rf_n_estimators = 40

    # -------------------------
    # Generate data
    # -------------------------
    xy = generate_interaction_dgp(
        n_total_obs=n_total_obs,
        p=p,
        rho=rho,
        beta_int=beta_int,
        beta3=beta3,
        sigma_eps=sigma_eps,
        seed=seed,
    )

    xy.to_csv(os.path.join(out_dir, "xy.csv"))

    subsets = AnatomySubsets.generate(
        index=xy.index,
        initial_window=initial_window,
        estimation_type=estimation_type,
        periods=periods,
        gap=gap,
    )

    print("\n" + "=" * 80)
    print("SIMULATION 2: Shapley vs APE vs OOB importance (linear vs tree)")
    print("=" * 80)
    print("DGP: y_{t+1} = %.2f*(x1_t*x2_t) + %.2f*x3_t + eps" % (beta_int, beta3))
    print("Irrelevant: x4..x%d" % p)
    print("n_total_obs=%d (usable rows=%d), initial_window=%d, n_periods=%d"
          % (n_total_obs, len(xy), initial_window, subsets.n_periods))
    print("Anatomy M=%d, RF trees=%d, delta(APE)=%0.2f" % (M, rf_n_estimators, delta))
    print("=" * 80 + "\n")

    # -------------------------
    # Estimate anatomy + compute TS-Shapley-VI and GPBSV
    # -------------------------
    anatomy_path = os.path.join(out_dir, "anatomy_sim2.bin")
    estimate_anatomy(
        xy=xy,
        subsets=subsets,
        n_iterations=M,
        save_path=anatomy_path,
        n_jobs=anatomy_jobs,
        base_seed_for_models=seed,
    )

    anatomy_obj = Anatomy.load(anatomy_path)

    metrics, df_forecasts, df_pbsv_rmse = compute_ts_shapley_vi_and_gpbsv(anatomy_obj)

    # Save raw anatomy outputs (optional but useful for debugging)
    df_forecasts.to_csv(os.path.join(out_dir, "df_forecasts_contribs.csv"))
    df_pbsv_rmse.to_csv(os.path.join(out_dir, "df_pbsv_rmse_decomp.csv"))

    # Save compact TS-Shapley-VI and GPBSV tables
    ts_vi_df = pd.DataFrame({m: metrics[m]["ts_shapley_vi"] for m in ["ols", "rf"]})
    gpbsv_df = pd.DataFrame({m: metrics[m]["gpbsv_rmse"] for m in ["ols", "rf"]})
    ts_vi_df.to_csv(os.path.join(out_dir, "TS_Shapley_VI.csv"))
    gpbsv_df.to_csv(os.path.join(out_dir, "GPBSV_RMSE.csv"))

    # -------------------------
    # Compute TS-APE and TS-OOB
    # -------------------------
    ts_ape_oob = compute_ts_ape_and_oob(
        xy=xy,
        subsets=subsets,
        delta=delta,
        rf_n_estimators=rf_n_estimators,
        rf_max_depth=rf_max_depth,
        rf_min_samples_leaf=rf_min_samples_leaf,
        rf_min_samples_split=rf_min_samples_split,
        rf_max_features=rf_max_features,
        seed=seed,
    )

    # Save TS-APE and TS-OOB as a single CSV
    feat_names = [c for c in xy.columns if c != "y"]
    ape_oob_df = pd.DataFrame(index=feat_names)
    for m in ["ols", "rf"]:
        ape_oob_df[m + "_APE"] = ts_ape_oob[m]["APE"]
        ape_oob_df[m + "_absAPE"] = ts_ape_oob[m]["absAPE"]
        ape_oob_df[m + "_OOB_MSE_increase"] = ts_ape_oob[m]["OOB_MSE_increase"]
    ape_oob_df.to_csv(os.path.join(out_dir, "TS_APE_and_OOB.csv"))

    # -------------------------
    # Build final tables (paper-style: include GPBSV + TS-Shapley-VI)
    # -------------------------
    true_effect_map = {f"x{i}": "irrelevant" for i in range(1, p + 1)}
    true_effect_map["x1"] = "interaction (x1*x2)"
    true_effect_map["x2"] = "interaction (x1*x2)"
    true_effect_map["x3"] = "linear"

    table_ols = build_model_table("ols", metrics["ols"], ts_ape_oob["ols"], true_effect_map)
    table_rf = build_model_table("rf", metrics["rf"], ts_ape_oob["rf"], true_effect_map)

    table_ols.to_csv(os.path.join(out_dir, "TABLE_ols.csv"))
    table_rf.to_csv(os.path.join(out_dir, "TABLE_rf.csv"))

    # -------------------------
    # Summary table (rank correlations etc.)
    # -------------------------
    summary_rows = []

    for m, table in [("ols", table_ols), ("rf", table_rf)]:
        rmse_total = float(metrics[m]["rmse"])

        # correlations (Spearman) between importance measures
        corr_vi_absape = spearman_corr(table["TS_Shapley_VI"], table["absAPE"])
        corr_vi_ape = spearman_corr(table["TS_Shapley_VI"], table["APE"])
        corr_vi_oob = spearman_corr(table["TS_Shapley_VI"], table["OOB_MSE_increase"]) if m == "rf" else np.nan

        # MAS (TS-Shapley-VI vs GPBSV)
        loss_type = MAS.LossType.LOWER_IS_BETTER
        mas_out = MAS(metrics[m]["ts_shapley_vi"], metrics[m]["gpbsv_rmse"], loss_type).compute(
            mas_type=MAS.MASType.IMPORTANCE_WEIGHTED,
            hypothesis_test=True,
            h0_alpha=0.50,
        )

        summary_rows.append(
            {
                "model": m,
                "RMSE_total": rmse_total,
                "MAS": float(mas_out.get("mas", np.nan)),
                "MAS_p_value": float(mas_out.get("mas_p_value", np.nan)),
                "Spearman(TSShapleyVI, absAPE)": corr_vi_absape,
                "Spearman(TSShapleyVI, APE)": corr_vi_ape,
                "Spearman(TSShapleyVI, OOB)": corr_vi_oob,
            }
        )

        # also save MAS json per model
        with open(os.path.join(out_dir, "MAS_%s.json" % m), "w") as f:
            json.dump(mas_out, f, indent=2)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(out_dir, "SUMMARY.csv"), index=False)

    # -------------------------
    # PRINT (console)
    # -------------------------
    pd.options.display.width = 0
    pd.options.display.max_columns = None
    pd.options.display.float_format = "{:.6f}".format

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(summary_df)

    print("\n" + "=" * 80)
    print("TABLE: OLS (sorted by GPBSV_RMSE ascending; most helpful at top)")
    print("=" * 80)
    print(table_ols)

    print("\n" + "=" * 80)
    print("TABLE: RF (sorted by GPBSV_RMSE ascending; most helpful at top)")
    print("=" * 80)
    print(table_rf)

    print("\nSaved outputs to: %s" % os.path.abspath(out_dir))
    print("Files: xy.csv, TS_Shapley_VI.csv, GPBSV_RMSE.csv, TS_APE_and_OOB.csv, TABLE_ols.csv, TABLE_rf.csv, SUMMARY.csv")


if __name__ == "__main__":
    main()
