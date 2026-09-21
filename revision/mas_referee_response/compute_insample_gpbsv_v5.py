"""
Compute empirical in-sample RMSE GPBSV objects for the CPI application.

This implements revision/submission_2/planning/in_sample_gpbsv_mas_plan_v5.md:
the new object is an in-sample GPBSV, while the existing out-of-sample GPBSV is
loaded, hashed, aligned, and used unchanged. This script never reconstructs or
replaces the saved OOS GPBSV.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_backend
from sklearn.preprocessing import MinMaxScaler


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
PACKAGE_ROOT = REPO_ROOT / "packages" / "anatomy"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from anatomy import (  # noqa: E402
    Anatomy,
    AnatomyAlgorithm,
    AnatomyModel,
    AnatomyModelCombination,
    AnatomyModelOutputTransformer,
    AnatomyModelProvider,
)


DEFAULT_RESULTS_DIR = REPO_ROOT / "Results" / "Updated CPI 3"
DEFAULT_MODEL_ROOT = REPO_ROOT / "Models" / "20241005_091334"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs" / "insample_gpbsv_v5"

HORIZONS = (1, 3, 6, 12)
BASE_MODEL_ORDER = ("rf_cv", "xgb_cv", "nn_deep", "nn_shallow", "enet", "pca")
REPORTED_MODEL_ORDER = (
    "rf_cv",
    "xgb_cv",
    "nn_comb",
    "enet",
    "pca",
    "lin_comb",
    "nonlin_comb",
    "all_comb",
)
OPERATIONAL_WEIGHTS: dict[str, dict[str, float]] = {
    "rf_cv": {"rf_cv": 1.0},
    "xgb_cv": {"xgb_cv": 1.0},
    "nn_comb": {"nn_deep": 0.5, "nn_shallow": 0.5},
    "enet": {"enet": 1.0},
    "pca": {"pca": 1.0},
    "lin_comb": {"enet": 0.5, "pca": 0.5},
    "nonlin_comb": {
        "rf_cv": 0.25,
        "xgb_cv": 0.25,
        "nn_deep": 0.25,
        "nn_shallow": 0.25,
    },
    "all_comb": {name: 1.0 / 6.0 for name in BASE_MODEL_ORDER},
}

OPERATIONAL_GROUPS = {
    model_key: [name for name, weight in weights.items() if weight > 0]
    for model_key, weights in OPERATIONAL_WEIGHTS.items()
}


@dataclass(frozen=True)
class AnatomyMetadata:
    horizon: int
    source_path: Path
    source_sha256: str
    n_periods: int
    y_name: str
    model_names: tuple[str, ...]
    columns: tuple[str, ...]
    permutations: np.ndarray


def repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def load_saved_oos(results_dir: Path, horizon: int) -> tuple[pd.DataFrame, Path, str]:
    artifact = results_dir / f"shapleys_h{horizon}_upd.bin"
    if not artifact.exists():
        raise FileNotFoundError(f"Missing saved OOS Shapley bundle: {artifact}")
    bundle = load_pickle(artifact)
    (pbsv_rmse, _pbsv_rmse_years), _shapleys, _oshapley_vi, _ishapley_vi = bundle
    if not isinstance(pbsv_rmse, pd.DataFrame):
        raise TypeError(f"Saved pbsv_rmse is not a DataFrame in {artifact}")
    return pbsv_rmse, artifact, sha256_file(artifact)


def load_anatomy_metadata(results_dir: Path, horizon: int) -> AnatomyMetadata:
    artifact = results_dir / f"anatomy_h{horizon}_upd.bin"
    if not artifact.exists():
        raise FileNotFoundError(f"Missing saved anatomy object: {artifact}")
    anatomy = Anatomy.load(str(artifact))
    permutations = np.asarray(anatomy._permutations).copy()
    columns = tuple(anatomy._columns)
    model_names = tuple(anatomy._model_names)
    return AnatomyMetadata(
        horizon=horizon,
        source_path=artifact,
        source_sha256=sha256_file(artifact),
        n_periods=int(anatomy._n_periods),
        y_name=str(anatomy._y_name),
        model_names=model_names,
        columns=columns,
        permutations=permutations,
    )


def load_saved_anatomy(results_dir: Path, horizon: int) -> Anatomy:
    artifact = results_dir / f"anatomy_h{horizon}_upd.bin"
    if not artifact.exists():
        raise FileNotFoundError(f"Missing saved anatomy object: {artifact}")
    return Anatomy.load(str(artifact))


def validate_permutations(permutations: np.ndarray, n_features: int) -> None:
    if permutations.ndim != 2:
        raise ValueError(f"Expected 2D permutation matrix, got {permutations.shape}")
    if permutations.shape[0] != n_features:
        raise ValueError(
            f"Permutation feature count {permutations.shape[0]} != {n_features}"
        )
    expected = np.arange(n_features)
    for m in range(permutations.shape[1]):
        if not np.array_equal(np.sort(permutations[:, m]), expected):
            raise ValueError(f"Permutation column {m} is not a permutation of 0..P-1")


def weight_vector(model_key: str, base_model_order: Iterable[str]) -> np.ndarray:
    weights = OPERATIONAL_WEIGHTS[model_key]
    vector = np.array([weights.get(name, 0.0) for name in base_model_order], dtype=float)
    if not np.isclose(vector.sum(), 1.0):
        raise ValueError(f"Weights for {model_key} do not sum to one: {vector}")
    return vector


def consolidate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Match Code/V002/iml_rev.py grouping: lags -> ar, x + x_ma3 -> x."""
    out = df.copy()
    lag_columns = [col for col in out.columns if col.startswith("y_t-")]
    if lag_columns:
        out["ar"] = out[lag_columns].sum(axis=1)

    grouped = []
    grouped_names = []
    for col in out.columns:
        if col.startswith("y_t-") or col.endswith("_ma3"):
            continue
        ma_col = f"{col}_ma3"
        if ma_col in out.columns:
            grouped.append(out[[col, ma_col]].sum(axis=1).rename(col))
        else:
            grouped.append(out[col].rename(col))
        grouped_names.append(col)

    if not grouped:
        return pd.DataFrame(index=out.index)
    result = pd.concat(grouped, axis=1)
    result.columns = grouped_names
    return result


def expected_grouped_columns(raw_columns: Iterable[str]) -> pd.Index:
    probe = pd.DataFrame(
        np.zeros((1, len(tuple(raw_columns)) + 1)),
        index=["probe"],
        columns=["base_contribution", *tuple(raw_columns)],
    )
    return consolidate_columns(probe).columns


def saved_oos_intake(
    *,
    pbsv_rmse: pd.DataFrame,
    oos_path: Path,
    oos_sha256: str,
    horizon: int,
    output_dir: Path,
    expected_rows: Iterable[str] = REPORTED_MODEL_ORDER,
    expected_columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    expected_rows = tuple(expected_rows)
    expected_columns_tuple = tuple(expected_columns) if expected_columns is not None else None
    row_match = tuple(pbsv_rmse.index) == expected_rows
    column_match = (
        True if expected_columns_tuple is None else tuple(pbsv_rmse.columns) == expected_columns_tuple
    )
    base_present = "base_contribution" in pbsv_rmse.columns
    intake_pass = bool(row_match and column_match and base_present)

    checks = pd.DataFrame([{
        "horizon": horizon,
        "saved_oos_source_path": str(oos_path),
        "saved_oos_sha256": oos_sha256,
        "saved_oos_model_rows": json.dumps(list(pbsv_rmse.index)),
        "saved_oos_grouped_columns": json.dumps(list(pbsv_rmse.columns)),
        "expected_model_rows": json.dumps(list(expected_rows)),
        "expected_grouped_columns": (
            "" if expected_columns_tuple is None else json.dumps(list(expected_columns_tuple))
        ),
        "row_name_match": row_match,
        "column_name_match": column_match,
        "base_contribution_present": base_present,
        "base_contribution_excluded_from_mas": True,
        "intake_pass": intake_pass,
    }])
    checks.to_csv(output_dir / f"saved_oos_alignment_h{horizon}_v5.csv", index=False)
    if not intake_pass:
        raise ValueError(
            f"Saved OOS intake failed for h={horizon}. "
            f"row_match={row_match}, column_match={column_match}, base_present={base_present}"
        )
    return checks


def write_structural_zero_mask(
    *,
    output_dir: Path,
    horizon: int,
    model_keys: Iterable[str],
    grouped_columns: Iterable[str],
) -> pd.DataFrame:
    predictors = [col for col in grouped_columns if col != "base_contribution"]
    rows = [
        {
            "horizon": horizon,
            "model": model_key,
            "predictor": predictor,
            "structural_zero": False,
            "source": "all_retained_no_direct_invariance_evidence",
        }
        for model_key in model_keys
        for predictor in predictors
    ]
    mask = pd.DataFrame(rows)
    mask.to_csv(output_dir / f"structural_zero_mask_h{horizon}.csv", index=False)
    return mask


def model_period_path(model_root: Path, horizon: int, period: int) -> Path:
    candidates = [
        model_root / f"cpiaucsl_h{horizon}" / f"{period}.bin",
        model_root / f"h{horizon}" / f"{period}.bin",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    archive = REPO_ROOT / "Models" / "20241005_091334.7z"
    hint = (
        f" Extract {repo_relative(archive)} and pass the extracted directory via --model-root."
        if archive.exists()
        else ""
    )
    raise FileNotFoundError(
        f"Could not find model file for h={horizon}, period={period} under {model_root}.{hint}"
    )


def validate_bundle_columns(
    *,
    bundle: dict,
    horizon: int,
    columns: tuple[str, ...],
    model_file: Path,
) -> pd.DataFrame:
    y_name = f"y_h{horizon}"
    train = bundle["train"]
    if not train.columns.is_unique:
        raise ValueError(f"Training columns are not unique in {model_file}")
    if y_name not in train.columns:
        raise ValueError(f"{y_name} missing from {model_file}")
    X_train = train.drop(y_name, axis=1)
    if y_name in X_train.columns:
        raise ValueError(f"Target column {y_name} leaked into features in {model_file}")
    if len(X_train.columns) != 247:
        raise ValueError(f"Expected 247 feature columns in {model_file}, got {len(X_train.columns)}")
    if tuple(X_train.columns) != columns:
        raise ValueError(f"Feature columns in {model_file} do not match saved anatomy columns")
    if not np.isfinite(X_train.to_numpy(dtype=float)).all():
        raise ValueError(f"Non-finite feature value in {model_file}")
    if not np.isfinite(train[y_name].to_numpy(dtype=float)).all():
        raise ValueError(f"Non-finite target value in {model_file}")

    if "xgb_cv" in bundle["models"]:
        booster_names = list(bundle["models"]["xgb_cv"][0].get_booster().feature_names)
        if booster_names != list(columns):
            raise ValueError(f"XGBoost booster feature names do not match saved columns in {model_file}")

    return X_train


def build_model_wrappers(bundle: dict, horizon: int) -> dict[str, AnatomyModel]:
    train = bundle["train"]
    y_name = f"y_h{horizon}"
    X_train = train.drop(y_name, axis=1)
    models = {}

    for model_name in BASE_MODEL_ORDER:
        if model_name in ("nn_deep", "nn_shallow"):
            fitted = bundle["models"][model_name][0]
            scaler = MinMaxScaler(feature_range=(-1, 1)).fit(X_train)

            def pred_fn(x: np.ndarray, fitted=fitted, scaler=scaler) -> np.ndarray:
                return np.array(fitted.predict(scaler.transform(x))).flatten()

        elif model_name == "pca":
            (pca_m, x_mean, x_std), fitted = bundle["models"][model_name]

            def pred_fn(
                x: np.ndarray,
                pca_m=pca_m,
                x_mean=x_mean,
                x_std=x_std,
                fitted=fitted,
            ) -> np.ndarray:
                x_pca = pca_m.transform((x - x_mean) / x_std)[:, : fitted.coef_.shape[0]]
                return np.array(fitted.predict(x_pca)).flatten()

        elif model_name == "enet":
            (x_mean, x_std), fitted = bundle["models"][model_name]

            def pred_fn(
                x: np.ndarray,
                x_mean=x_mean,
                x_std=x_std,
                fitted=fitted,
            ) -> np.ndarray:
                return np.array(fitted.predict((x - x_mean) / x_std)).flatten()

        elif model_name == "xgb_cv":
            fitted = bundle["models"][model_name][0]
            feature_names = list(fitted.get_booster().feature_names)

            def pred_fn(
                x: np.ndarray,
                fitted=fitted,
                feature_names=feature_names,
            ) -> np.ndarray:
                x_df = pd.DataFrame(x, columns=feature_names)
                return np.array(fitted.predict(x_df)).flatten()

        else:
            fitted = bundle["models"][model_name][0]

            def pred_fn(x: np.ndarray, fitted=fitted) -> np.ndarray:
                return np.array(fitted.predict(x)).flatten()

        models[model_name] = AnatomyModel(pred_fn)

    return models


def choose_positions(
    *,
    n_rows: int,
    mode: str,
    rows_per_window: int | None,
    rng: np.random.Generator,
) -> np.ndarray:
    if mode == "exact-stream":
        return np.arange(n_rows, dtype=int)
    if rows_per_window is None or rows_per_window <= 0:
        raise ValueError("--rows-per-window must be positive in sampled-stream mode")
    k = min(int(rows_per_window), n_rows)
    return np.sort(rng.choice(n_rows, size=k, replace=False))


def select_audit_periods(n_periods: int, count: int) -> list[int]:
    count = max(0, min(int(count), int(n_periods)))
    if count == 0:
        return []
    if count == 1:
        return [0]
    return sorted({int(round(x)) for x in np.linspace(0, n_periods - 1, count)})


def period_row_slice(anatomy: Anatomy, period: int) -> slice:
    n_periods = int(anatomy._n_periods)
    n_rows = len(anatomy._xy_test)
    if n_rows % n_periods != 0:
        raise ValueError(f"Cannot infer rows per period: {n_rows} rows, {n_periods} periods")
    rows_per_period = n_rows // n_periods
    start = int(period) * rows_per_period
    return slice(start, start + rows_per_period)


def run_forecast_parity_gate(
    *,
    args: argparse.Namespace,
    horizon: int,
    metadata: AnatomyMetadata,
) -> pd.DataFrame:
    anatomy = load_saved_anatomy(args.results_dir, horizon)
    periods = select_audit_periods(metadata.n_periods, args.forecast_parity_periods)
    rows = []
    max_error = 0.0

    for period in periods:
        model_file = model_period_path(args.model_root, horizon, period)
        bundle = load_pickle(model_file)
        X_train = validate_bundle_columns(
            bundle=bundle,
            horizon=horizon,
            columns=metadata.columns,
            model_file=model_file,
        )
        del X_train
        y_name = f"y_h{horizon}"
        test = bundle["test"]
        if tuple(test.drop(y_name, axis=1).columns) != metadata.columns:
            raise ValueError(f"Test columns in {model_file} do not match saved anatomy columns")

        row_slice = period_row_slice(anatomy, period)
        saved_test = anatomy._xy_test.iloc[row_slice]
        if not saved_test.equals(test):
            raise ValueError(f"Saved anatomy test rows differ from model bundle test rows for {model_file}")

        wrappers = build_model_wrappers(bundle, horizon)
        X_test = test.drop(y_name, axis=1)
        for model_index, model_name in enumerate(BASE_MODEL_ORDER):
            pred = wrappers[model_name].predict(X_test.to_numpy())
            saved_y_hat = anatomy._y_hats[model_index, row_slice]
            full_cells = anatomy._Y[model_index, row_slice, -1, :, :]
            y_hat_error = float(np.max(np.abs(pred - saved_y_hat)))
            full_error = float(np.max(np.abs(full_cells - pred[:, None, None])))
            max_error = max(max_error, y_hat_error, full_error)
            rows.append({
                "horizon": horizon,
                "period": period,
                "model": model_name,
                "n_test_rows": len(test),
                "max_abs_error_y_hat": y_hat_error,
                "max_abs_error_full_coalition": full_error,
                "pass": bool(
                    np.allclose(pred, saved_y_hat, atol=args.atol, rtol=args.rtol)
                    and np.allclose(full_cells, pred[:, None, None], atol=args.atol, rtol=args.rtol)
                ),
            })

    report = pd.DataFrame(rows)
    report.to_csv(args.output_dir / f"forecast_parity_h{horizon}_v5.csv", index=False)
    if not report.empty and not bool(report["pass"].all()):
        raise ValueError(f"Forecast parity gate failed for h={horizon}; max error={max_error}")
    print(f"h={horizon}: forecast parity gate passed on {len(periods)} periods")
    return report


def explain_from_value_grid(
    value_grid: np.ndarray,
    permutations: np.ndarray,
    *,
    atol: float,
    rtol: float,
) -> tuple[np.ndarray, float, float, dict[str, float]]:
    if value_grid.ndim != 3:
        raise ValueError(f"Expected value grid (P+1,M,2), got {value_grid.shape}")

    empty_values = value_grid[0]
    full_values = value_grid[-1]
    empty_ref = float(empty_values[0, 0])
    full_ref = float(full_values[0, 0])
    empty_max_diff = float(np.max(np.abs(empty_values - empty_ref)))
    full_max_diff = float(np.max(np.abs(full_values - full_ref)))
    if not np.allclose(empty_values, empty_ref, atol=atol, rtol=rtol):
        raise ValueError(f"Empty coalition is not invariant; max diff={empty_max_diff}")
    if not np.allclose(full_values, full_ref, atol=atol, rtol=rtol):
        raise ValueError(f"Full coalition is not invariant; max diff={full_max_diff}")

    marginal = np.diff(value_grid, axis=0)
    phi = np.empty_like(marginal)
    for m in range(permutations.shape[1]):
        phi[permutations[:, m], m, 0] = marginal[:, m, 0]
        phi[permutations[:, m][::-1], m, 1] = marginal[:, m, 1]
    phi_raw = phi.mean(axis=2).mean(axis=1)
    efficiency_residual = float(empty_ref + phi_raw.sum() - full_ref)
    if not np.isclose(efficiency_residual, 0.0, atol=atol, rtol=rtol):
        raise ValueError(f"Efficiency check failed; residual={efficiency_residual}")

    return phi_raw, empty_ref, full_ref, {
        "empty_max_diff": empty_max_diff,
        "full_max_diff": full_max_diff,
        "efficiency_residual": efficiency_residual,
    }


def window_mse_grids(
    *,
    model_file: Path,
    bundle: dict,
    horizon: int,
    columns: tuple[str, ...],
    permutations: np.ndarray,
    positions: np.ndarray,
    model_keys: tuple[str, ...],
    chunk_size: int,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    y_name = f"y_h{horizon}"
    train = bundle["train"]
    X_train = validate_bundle_columns(
        bundle=bundle,
        horizon=horizon,
        columns=columns,
        model_file=model_file,
    )
    if positions.size == 0:
        raise ValueError(f"No evaluation rows selected for {model_file}")

    models = build_model_wrappers(bundle, horizon)
    sse = {
        key: np.zeros((len(columns) + 1, permutations.shape[1], 2), dtype=np.float64)
        for key in model_keys
    }
    y_eval = train[y_name].iloc[positions].to_numpy(dtype=float)
    X_eval = X_train.iloc[positions]

    for start in range(0, len(positions), chunk_size):
        stop = min(start + chunk_size, len(positions))
        xs_chunk = X_eval.iloc[start:stop]
        y_chunk = y_eval[start:stop]
        base_outputs = []
        for model_name in BASE_MODEL_ORDER:
            base_outputs.append(
                AnatomyAlgorithm.evaluate_permuted_orders(
                    model=models[model_name],
                    X=X_train,
                    xs=xs_chunk,
                    permutations=permutations,
                    subsample=1.0,
                )
            )
        y_base = np.stack(base_outputs, axis=0)
        for model_key in model_keys:
            weights = weight_vector(model_key, BASE_MODEL_ORDER)
            y_group = np.tensordot(weights, y_base, axes=(0, 0))
            errors2 = (y_chunk[:, None, None, None] - y_group) ** 2
            sse[model_key] += errors2.sum(axis=0)

    mse = {key: value / float(len(positions)) for key, value in sse.items()}
    selected = pd.DataFrame({
        "row_position": positions,
        "row_label": [str(x) for x in train.index[positions]],
        "window_n_rows": len(train),
        "row_inclusion_probability": len(positions) / float(len(train)),
    })
    return mse, selected


def run_provider_audit_gate(
    *,
    args: argparse.Namespace,
    horizon: int,
    metadata: AnatomyMetadata,
    model_keys: tuple[str, ...],
) -> pd.DataFrame:
    periods = select_audit_periods(metadata.n_periods, args.provider_audit_periods)
    rows = []
    y_name = f"y_h{horizon}"

    for period in periods:
        model_file = model_period_path(args.model_root, horizon, period)
        bundle = load_pickle(model_file)
        X_train = validate_bundle_columns(
            bundle=bundle,
            horizon=horizon,
            columns=metadata.columns,
            model_file=model_file,
        )
        rng = np.random.default_rng(stable_seed(args.seed, "provider-audit", horizon, period))
        positions = choose_positions(
            n_rows=len(bundle["train"]),
            mode="sampled-stream",
            rows_per_window=args.provider_audit_rows,
            rng=rng,
        )
        mse, _selected = window_mse_grids(
            model_file=model_file,
            bundle=bundle,
            horizon=horizon,
            columns=metadata.columns,
            permutations=metadata.permutations,
            positions=positions,
            model_keys=model_keys,
            chunk_size=int(args.chunk_size),
        )

        wrappers = build_model_wrappers(bundle, horizon)
        test_rows = bundle["train"].iloc[positions].copy()

        def mapper(key: AnatomyModelProvider.PeriodKey) -> AnatomyModelProvider.PeriodValue:
            return AnatomyModelProvider.PeriodValue(
                train=bundle["train"],
                test=test_rows,
                model=wrappers[key.model_name],
            )

        provider = AnatomyModelProvider(
            n_periods=1,
            n_features=len(metadata.columns),
            model_names=list(BASE_MODEL_ORDER),
            y_name=y_name,
            provider_fn=mapper,
        )
        anatomy = Anatomy(provider=provider, n_iterations=metadata.permutations.shape[1])
        anatomy._permutations = metadata.permutations.copy()
        anatomy.precompute(n_jobs=int(args.provider_audit_jobs))

        def transform(y_hat, y):
            return np.sqrt(np.mean((y - y_hat) ** 2))

        package = anatomy.explain(
            model_sets=AnatomyModelCombination(
                groups={key: OPERATIONAL_GROUPS[key] for key in model_keys}
            ),
            transformer=AnatomyModelOutputTransformer(transform=transform),
        )

        for model_key in model_keys:
            stream_phi, stream_base, _stream_full, _diagnostics = explain_from_value_grid(
                np.sqrt(mse[model_key]),
                metadata.permutations,
                atol=args.atol,
                rtol=args.rtol,
            )
            stream_row = pd.Series(
                np.r_[stream_base, stream_phi],
                index=["base_contribution", *metadata.columns],
            )
            package_row = package.loc[model_key].iloc[0].reindex(stream_row.index)
            max_abs_diff = float(np.max(np.abs(stream_row.to_numpy() - package_row.to_numpy())))
            passed = bool(np.allclose(stream_row, package_row, atol=args.atol, rtol=args.rtol))
            rows.append({
                "horizon": horizon,
                "period": period,
                "model": model_key,
                "n_rows": len(positions),
                "max_abs_diff": max_abs_diff,
                "pass": passed,
            })

    report = pd.DataFrame(rows)
    report.to_csv(args.output_dir / f"provider_audit_h{horizon}_v5.csv", index=False)
    if not report.empty and not bool(report["pass"].all()):
        raise ValueError(
            f"Provider audit failed for h={horizon}; max diff={report['max_abs_diff'].max()}"
        )
    print(f"h={horizon}: provider audit passed on {len(periods)} periods")
    return report


def _window_cache_path(
    cache_dir: Path,
    *,
    horizon: int,
    rep: int,
    period: int,
    rows_per_window: int | None,
    seed: int,
    mode: str,
) -> Path:
    """Per-window cache filename, keyed by every parameter that changes the result."""
    k = "all" if (mode == "exact-stream" or rows_per_window is None) else int(rows_per_window)
    return cache_dir / f"h{horizon}_rep{rep}_period{period}_k{k}_seed{seed}_{mode}.pkl"


def _load_window_cache(
    path: Path,
    *,
    model_keys: tuple[str, ...],
    n_columns: int,
    m: int,
) -> tuple[dict[str, np.ndarray], pd.DataFrame] | None:
    """Return (mse, selected) from a window cache file, or None if absent/stale."""
    if not path.exists():
        return None
    try:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
    except Exception:
        return None
    if payload.get("schema") != "window_cache_v1":
        return None
    if payload.get("n_columns") != n_columns or payload.get("m") != m:
        return None
    mse = payload.get("mse")
    selected = payload.get("selected")
    if not isinstance(mse, dict) or selected is None:
        return None
    if not all(key in mse for key in model_keys):
        return None
    return {key: mse[key] for key in model_keys}, selected


def _write_window_cache(
    cache_dir: Path,
    *,
    horizon: int,
    rep: int,
    period: int,
    rows_per_window: int | None,
    seed: int,
    mode: str,
    mse: dict[str, np.ndarray],
    selected: pd.DataFrame,
    n_columns: int,
    m: int,
    model_keys: tuple[str, ...],
) -> None:
    """Atomically persist one window's grids so a crashed run can resume."""
    path = _window_cache_path(
        cache_dir, horizon=horizon, rep=rep, period=period,
        rows_per_window=rows_per_window, seed=seed, mode=mode,
    )
    payload = {
        "schema": "window_cache_v1",
        "horizon": horizon,
        "rep": rep,
        "period": period,
        "rows_per_window": rows_per_window,
        "seed": seed,
        "mode": mode,
        "n_columns": n_columns,
        "m": m,
        "model_keys": list(model_keys),
        "mse": mse,
        "selected": selected,
    }
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as handle:
        pickle.dump(payload, handle)
    tmp.replace(path)


def _compute_window_task(
    period: int,
    *,
    model_root: Path,
    horizon: int,
    columns: tuple[str, ...],
    permutations: np.ndarray,
    model_keys: tuple[str, ...],
    chunk_size: int,
    seed: int,
    rep: int,
    mode: str,
    rows_per_window: int | None,
    cache_dir: Path | None = None,
) -> tuple[int, dict[str, np.ndarray], pd.DataFrame]:
    """Compute one window's MSE grids.

    Top-level (picklable) so the window loop can be parallelized across
    processes. Row positions are drawn from a window-specific deterministic
    seed, so the selected rows and the returned grids are independent of
    execution order or the number of workers. When cache_dir is given the
    result is persisted atomically before returning, so a crashed run resumes.
    """
    model_file = model_period_path(model_root, horizon, int(period))
    bundle = load_pickle(model_file)
    n_rows = len(bundle["train"])
    rng = np.random.default_rng(stable_seed(seed, "row-sample", horizon, rep, period))
    positions = choose_positions(
        n_rows=n_rows,
        mode=mode,
        rows_per_window=rows_per_window,
        rng=rng,
    )
    mse, selected = window_mse_grids(
        model_file=model_file,
        bundle=bundle,
        horizon=horizon,
        columns=columns,
        permutations=permutations,
        positions=positions,
        model_keys=model_keys,
        chunk_size=chunk_size,
    )
    if cache_dir is not None:
        _write_window_cache(
            cache_dir,
            horizon=horizon,
            rep=rep,
            period=int(period),
            rows_per_window=rows_per_window,
            seed=seed,
            mode=mode,
            mse=mse,
            selected=selected,
            n_columns=len(columns) + 1,
            m=permutations.shape[1],
            model_keys=model_keys,
        )
    return int(period), mse, selected


def compute_horizon(args: argparse.Namespace, horizon: int) -> None:
    output_dir = ensure_dir(args.output_dir)
    pbsv_rmse, oos_path, oos_sha = load_saved_oos(args.results_dir, horizon)

    if args.mode == "dry-run" and not args.load_anatomy:
        saved_oos_intake(
            pbsv_rmse=pbsv_rmse,
            oos_path=oos_path,
            oos_sha256=oos_sha,
            horizon=horizon,
            output_dir=output_dir,
            expected_columns=pbsv_rmse.columns,
        )
        print(f"h={horizon}: saved OOS intake passed without loading anatomy")
        return

    metadata = load_anatomy_metadata(args.results_dir, horizon)
    if metadata.model_names != BASE_MODEL_ORDER:
        raise ValueError(
            f"Saved anatomy model order {metadata.model_names} != {BASE_MODEL_ORDER}"
        )
    validate_permutations(metadata.permutations, len(metadata.columns))
    if metadata.permutations.shape[1] != 100:
        raise ValueError(f"Expected M=100 permutations, got {metadata.permutations.shape[1]}")

    grouped_columns = expected_grouped_columns(metadata.columns)
    saved_oos_intake(
        pbsv_rmse=pbsv_rmse,
        oos_path=oos_path,
        oos_sha256=oos_sha,
        horizon=horizon,
        output_dir=output_dir,
        expected_columns=grouped_columns,
    )
    write_structural_zero_mask(
        output_dir=output_dir,
        horizon=horizon,
        model_keys=args.models,
        grouped_columns=grouped_columns,
    )

    if args.mode == "dry-run":
        print(f"h={horizon}: dry-run passed with anatomy metadata")
        return

    model_root = args.model_root
    if not model_root.exists():
        archive = REPO_ROOT / "Models" / "20241005_091334.7z"
        raise FileNotFoundError(
            f"Model root does not exist: {model_root}. "
            f"Extract {repo_relative(archive)} and pass --model-root to the extracted root."
        )

    model_keys = tuple(args.models)
    unknown = sorted(set(model_keys) - set(REPORTED_MODEL_ORDER))
    if unknown:
        raise ValueError(f"Unknown model keys: {unknown}")

    n_periods_total = metadata.n_periods
    if args.max_windows is not None and not args.allow_window_subset:
        raise ValueError(
            "--max-windows creates a deterministic prefix pilot, not the full equal-window target. "
            "Use --allow-window-subset to run this explicitly as a pilot."
        )

    forecast_parity_report = None
    provider_audit_report = None
    if not args.skip_forecast_parity:
        forecast_parity_report = run_forecast_parity_gate(
            args=args,
            horizon=horizon,
            metadata=metadata,
        )
    if not args.skip_provider_audit:
        provider_audit_report = run_provider_audit_gate(
            args=args,
            horizon=horizon,
            metadata=metadata,
            model_keys=model_keys,
        )
    if args.mode == "provider-audit":
        return

    window_ids = np.arange(n_periods_total, dtype=int)
    if args.max_windows is not None:
        window_ids = window_ids[: int(args.max_windows)]
    if window_ids.size == 0:
        raise ValueError("No windows selected")

    sampling_replicates = 1 if args.mode == "exact-stream" else int(args.sampling_replicates)
    if sampling_replicates <= 0:
        raise ValueError("--sampling-replicates must be positive")

    total_mse = {
        key: np.zeros((len(metadata.columns) + 1, metadata.permutations.shape[1], 2), dtype=np.float64)
        for key in model_keys
    }
    replicate_records = []
    selected_rows_records = []

    checkpoint = not args.no_checkpoint
    cache_dir = ensure_dir(output_dir / f"_window_cache_h{horizon}") if checkpoint else None

    for rep in range(sampling_replicates):
        rep_mse = {
            key: np.zeros_like(total_mse[key])
            for key in model_keys
        }
        # Resume: load windows already cached from a previous run, dispatch the rest.
        # Cached grids are bit-identical to recomputed ones (same window-specific
        # seed -> same rows), so a resumed run equals a fresh run.
        cached_results = []
        pending_periods = []
        for period in window_ids:
            loaded = None
            if checkpoint:
                loaded = _load_window_cache(
                    _window_cache_path(
                        cache_dir, horizon=horizon, rep=rep, period=int(period),
                        rows_per_window=args.rows_per_window, seed=int(args.seed), mode=args.mode,
                    ),
                    model_keys=model_keys,
                    n_columns=len(metadata.columns) + 1,
                    m=metadata.permutations.shape[1],
                )
            if loaded is not None:
                mse_cached, selected_cached = loaded
                cached_results.append((int(period), mse_cached, selected_cached))
            else:
                pending_periods.append(int(period))
        if cached_results:
            print(
                f"h={horizon} rep={rep + 1}/{sampling_replicates}: resuming, "
                f"{len(cached_results)}/{window_ids.size} windows already cached"
            )

        window_tasks = [
            delayed(_compute_window_task)(
                int(period),
                model_root=model_root,
                horizon=horizon,
                columns=metadata.columns,
                permutations=metadata.permutations,
                model_keys=model_keys,
                chunk_size=int(args.chunk_size),
                seed=int(args.seed),
                rep=rep,
                mode=args.mode,
                rows_per_window=args.rows_per_window,
                cache_dir=cache_dir,
            )
            for period in pending_periods
        ]
        if not window_tasks:
            computed = []
        elif int(args.n_jobs) == 1:
            computed = [fn(*a, **kw) for fn, a, kw in window_tasks]
        else:
            # Each worker runs single-threaded (inner_max_num_threads=1) so the
            # window-level parallelism does not oversubscribe RF/BLAS threads.
            with parallel_backend("loky", inner_max_num_threads=1):
                computed = Parallel(n_jobs=int(args.n_jobs), verbose=10)(window_tasks)
        # Aggregate in ascending period order so the equal-window 1/N reduction is
        # taken in a fixed, n_jobs-independent order. Row sampling (window-specific
        # seed) and weighting are also n_jobs-independent, so the estimand does not
        # depend on worker count or on how many windows were resumed from cache.
        # Workers use single-threaded BLAS, so ensemble forecasts can differ from
        # the n_jobs=1 path only by ~1e-12 thread-rounding noise (far inside the
        # efficiency tolerance); n_jobs=1 with no cache reproduces the prior code.
        results = cached_results + list(computed)
        results.sort(key=lambda item: item[0])
        for period, mse, selected in results:
            for key in model_keys:
                rep_mse[key] += mse[key] / float(window_ids.size)
            selected.insert(0, "period", int(period))
            selected.insert(0, "replicate", rep)
            selected_rows_records.append(selected)
        print(
            f"h={horizon} rep={rep + 1}/{sampling_replicates}: "
            f"{len(results)}/{n_periods_total} windows done "
            f"({len(cached_results)} cached, {len(window_tasks)} computed, n_jobs={args.n_jobs})"
        )

        for key in model_keys:
            total_mse[key] += rep_mse[key] / float(sampling_replicates)

        if sampling_replicates > 1:
            rep_rows = []
            for key in model_keys:
                rmse_grid = np.sqrt(rep_mse[key])
                phi, base, full, diagnostics = explain_from_value_grid(
                    rmse_grid,
                    metadata.permutations,
                    atol=args.atol,
                    rtol=args.rtol,
                )
                row = {"replicate": rep, "model": key, "base_contribution": base, "full_loss": full}
                row.update(diagnostics)
                row.update(dict(zip(metadata.columns, phi)))
                rep_rows.append(row)
            replicate_records.extend(rep_rows)

    raw_rows = []
    check_rows = []
    for key in model_keys:
        rmse_grid = np.sqrt(total_mse[key])
        phi, base, full, diagnostics = explain_from_value_grid(
            rmse_grid,
            metadata.permutations,
            atol=args.atol,
            rtol=args.rtol,
        )
        raw_rows.append(pd.Series(
            np.r_[base, phi],
            index=["base_contribution", *metadata.columns],
            name=key,
        ))
        check_rows.append({
            "horizon": horizon,
            "model": key,
            "base_contribution": base,
            "full_loss": full,
            **diagnostics,
        })

    raw = pd.DataFrame(raw_rows)
    grouped = consolidate_columns(raw)
    checks = pd.DataFrame(check_rows)
    selected_rows = pd.concat(selected_rows_records, ignore_index=True)
    replicate_df = pd.DataFrame(replicate_records)

    raw.to_csv(output_dir / f"insample_gpbsv_h{horizon}_raw.csv")
    grouped.to_csv(output_dir / f"insample_gpbsv_h{horizon}_grouped.csv")
    checks.to_csv(output_dir / f"insample_gpbsv_h{horizon}_checks.csv", index=False)
    selected_rows.to_csv(output_dir / f"insample_gpbsv_h{horizon}_selected_rows.csv", index=False)
    if not replicate_df.empty:
        replicate_df.to_parquet(output_dir / f"insample_gpbsv_h{horizon}_sampling_replicates.parquet")

    payload = {
        "schema_version": "insample_gpbsv_v5",
        "loss_game_id": "equal_window_global_rmse",
        "loss_formula": "sqrt(mean_i(mean_t(error^2)))",
        "horizon": horizon,
        "mode": args.mode,
        "n_windows_total": n_periods_total,
        "n_windows_used": int(window_ids.size),
        "pilot_biased_prefix": bool(args.max_windows is not None),
        "window_ids": window_ids.tolist(),
        "row_sampling_design": {
            "rows_per_window": args.rows_per_window,
            "sampling_replicates": sampling_replicates,
            "seed": args.seed,
            "n_jobs": int(args.n_jobs),
            "checkpoint_dir": (str(cache_dir) if cache_dir is not None else None),
        },
        "forecast_parity_max_abs_error": (
            None if forecast_parity_report is None or forecast_parity_report.empty
            else float(forecast_parity_report[
                ["max_abs_error_y_hat", "max_abs_error_full_coalition"]
            ].max().max())
        ),
        "provider_audit_max_abs_diff": (
            None if provider_audit_report is None or provider_audit_report.empty
            else float(provider_audit_report["max_abs_diff"].max())
        ),
        "structural_zero_policy": "all retained; no direct invariance evidence in this packet",
        "m": int(metadata.permutations.shape[1]),
        "antithetic": True,
        "anatomy_source": str(metadata.source_path),
        "anatomy_sha256": metadata.source_sha256,
        "saved_oos_artifact_source": str(oos_path),
        "saved_oos_artifact_sha256": oos_sha,
        "oos_artifact_policy": "frozen_saved_object",
        "oos_intake_policy": "load_hash_align_use_unchanged",
        "columns": list(metadata.columns),
        "base_model_order": list(BASE_MODEL_ORDER),
        "reported_model_order": list(model_keys),
        "ensemble_weights": OPERATIONAL_WEIGHTS,
        "is_gpbsv_rmse_raw": raw,
        "is_gpbsv_rmse_grouped": grouped,
        "checks": checks,
    }
    with (output_dir / f"insample_gpbsv_h{horizon}_v5.bin").open("wb") as handle:
        pickle.dump(payload, handle)
    print(f"h={horizon}: wrote {repo_relative(output_dir)}")


def run_self_tests() -> None:
    permutations = np.array([[0], [1]])
    value_grid = np.zeros((3, 1, 2), dtype=float)
    value_grid[:, 0, 0] = [10.0, 7.0, 5.0]
    value_grid[:, 0, 1] = [10.0, 8.0, 5.0]
    phi, base, full, diagnostics = explain_from_value_grid(
        value_grid,
        permutations,
        atol=1e-12,
        rtol=1e-12,
    )
    assert np.allclose(phi, [-3.0, -2.0])
    assert np.isclose(base, 10.0)
    assert np.isclose(full, 5.0)
    assert np.isclose(diagnostics["efficiency_residual"], 0.0)

    weights = weight_vector("nonlin_comb", BASE_MODEL_ORDER)
    assert np.isclose(weights.sum(), 1.0)
    assert np.allclose(weights[:4], [0.25, 0.25, 0.25, 0.25])

    # Checkpoint cache round-trip and staleness guards.
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        cdir = Path(tmp)
        fake_mse = {
            "rf_cv": np.arange(6, dtype=float).reshape(3, 1, 2),
            "enet": np.ones((3, 1, 2), dtype=float),
        }
        fake_sel = pd.DataFrame({"row_position": [0, 1], "row_label": ["a", "b"]})
        _write_window_cache(
            cdir, horizon=1, rep=0, period=7, rows_per_window=8, seed=123,
            mode="sampled-stream", mse=fake_mse, selected=fake_sel,
            n_columns=3, m=1, model_keys=("rf_cv", "enet"),
        )
        cpath = _window_cache_path(
            cdir, horizon=1, rep=0, period=7, rows_per_window=8, seed=123, mode="sampled-stream",
        )
        loaded = _load_window_cache(cpath, model_keys=("rf_cv", "enet"), n_columns=3, m=1)
        assert loaded is not None
        mse_back, sel_back = loaded
        assert np.array_equal(mse_back["rf_cv"], fake_mse["rf_cv"])
        assert sel_back.equals(fake_sel)
        # A subset of the cached models is reusable.
        assert _load_window_cache(cpath, model_keys=("enet",), n_columns=3, m=1) is not None
        # Staleness guards: wrong column count, wrong M, or a missing model -> miss.
        assert _load_window_cache(cpath, model_keys=("rf_cv",), n_columns=99, m=1) is None
        assert _load_window_cache(cpath, model_keys=("rf_cv",), n_columns=3, m=2) is None
        assert _load_window_cache(cpath, model_keys=("xgb_cv",), n_columns=3, m=1) is None
        # A different (k, seed, mode) is a different file -> absent (no stale reuse).
        other = _window_cache_path(
            cdir, horizon=1, rep=0, period=7, rows_per_window=4, seed=123, mode="sampled-stream",
        )
        assert _load_window_cache(other, model_keys=("rf_cv",), n_columns=3, m=1) is None

    print("Self-tests passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute v5 in-sample RMSE GPBSV against frozen saved OOS GPBSV."
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--horizons", type=int, nargs="+", default=list(HORIZONS))
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(REPORTED_MODEL_ORDER),
        choices=list(REPORTED_MODEL_ORDER),
    )
    parser.add_argument(
        "--mode",
        choices=["dry-run", "sampled-stream", "exact-stream", "provider-audit"],
        default="dry-run",
    )
    parser.add_argument(
        "--load-anatomy",
        action="store_true",
        help="In dry-run mode, also load the large saved anatomy object to check permutations and columns.",
    )
    parser.add_argument("--rows-per-window", type=int, default=32)
    parser.add_argument("--sampling-replicates", type=int, default=1)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument(
        "--allow-window-subset",
        action="store_true",
        help="Permit --max-windows as an explicit deterministic-prefix pilot.",
    )
    parser.add_argument("--chunk-size", type=int, default=1)
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Worker processes over the window loop (1 = sequential, unchanged). "
             "Each worker is single-threaded; on this 16-core box use ~16. "
             "Results are bit-identical to n-jobs=1.",
    )
    parser.add_argument(
        "--no-checkpoint",
        action="store_true",
        help="Disable per-window checkpoint caching/resume (default: enabled). "
             "With it enabled, a crashed run resumes from the last completed window.",
    )
    parser.add_argument("--seed", type=int, default=20260618)
    parser.add_argument("--atol", type=float, default=1e-8)
    parser.add_argument("--rtol", type=float, default=1e-8)
    parser.add_argument(
        "--forecast-parity-periods",
        type=int,
        default=3,
        help="Number of windows checked against saved anatomy forecasts before streaming.",
    )
    parser.add_argument(
        "--skip-forecast-parity",
        action="store_true",
        help="Skip the saved-forecast parity gate. Intended only for debugging.",
    )
    parser.add_argument(
        "--provider-audit-periods",
        type=int,
        default=1,
        help="Number of windows used in the package-vs-streaming provider audit.",
    )
    parser.add_argument(
        "--provider-audit-rows",
        type=int,
        default=1,
        help="Training rows per audited window in provider-audit mode.",
    )
    parser.add_argument(
        "--provider-audit-jobs",
        type=int,
        default=1,
        help="n_jobs passed to Anatomy.precompute during provider audit.",
    )
    parser.add_argument(
        "--skip-provider-audit",
        action="store_true",
        help="Skip the package provider-vs-streaming audit gate. Intended only for debugging.",
    )
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_tests()
        if args.mode == "dry-run" and not args.horizons:
            return
    if args.mode == "sampled-stream" and args.rows_per_window <= 0:
        raise ValueError("--rows-per-window must be positive in sampled-stream mode")
    if args.mode == "provider-audit" and args.provider_audit_rows <= 0:
        raise ValueError("--provider-audit-rows must be positive in provider-audit mode")
    for horizon in args.horizons:
        compute_horizon(args, int(horizon))


if __name__ == "__main__":
    main()
