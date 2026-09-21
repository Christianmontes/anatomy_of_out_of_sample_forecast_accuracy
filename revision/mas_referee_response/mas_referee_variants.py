"""
Standalone MAS appendix variants for the referee response.

This script intentionally does not import or modify Code/V002/iml_rev.py. It reads
the latest saved Shapley/GPBSV bundles and writes isolated appendix outputs under
this revision folder by default.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_RESULTS_DIR = REPO_ROOT / "Results" / "Updated CPI 3"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs"
DEFAULT_INSAMPLE_GPBSV_DIR = SCRIPT_DIR / "outputs" / "insample_gpbsv_v5"

HORIZONS = (1, 3, 6, 12)
MODEL_KEYS = (
    "pca",
    "enet",
    "rf_cv",
    "xgb_cv",
    "nn_comb",
    "lin_comb",
    "nonlin_comb",
    "all_comb",
)
MODEL_LABELS = {
    "pca": "PCA",
    "enet": "ENet",
    "rf_cv": "RF",
    "xgb_cv": "XGBoost",
    "nn_comb": "Neural net",
    "lin_comb": "Linear combination",
    "nonlin_comb": "Nonlinear combination",
    "all_comb": "All models combination",
}
COMPARISON_LABELS = {
    "is_shapley_vs_oos_gpbsv": "Current: IS Shapley VI vs OOS GPBSV",
    "is_shapley_vs_oos_shapley": "Check 1: IS Shapley VI vs OOS Shapley VI",
    "oos_shapley_vs_oos_gpbsv": "Check 2: OOS Shapley VI vs OOS GPBSV",
    "oos_shapley_vs_oos_gpbsv_is_weighted": (
        "Sensitivity: OOS Shapley VI vs OOS GPBSV, IS weights"
    ),
    "is_gpbsv_abs_vs_saved_oos_gpbsv_signed": (
        "IS GPBSV robustness: abs(IS GPBSV) vs saved OOS GPBSV"
    ),
    "is_gpbsv_signed_vs_saved_oos_gpbsv_signed": (
        "IS GPBSV concordance: signed IS GPBSV vs saved OOS GPBSV"
    ),
}


@dataclass(frozen=True)
class MASResult:
    horizon: int
    model: str
    model_label: str
    comparison: str
    comparison_label: str
    weighting_source: str
    null_type: str
    mas: float
    p_value: float
    msdr: float
    expected_msdr: float
    null_alpha: float
    n_sims: int
    seed: int
    n_predictors: int
    n_good_gpbsv: Optional[int]
    n_bad_gpbsv: Optional[int]
    n_dropped_base: int
    n_dropped_zero_weight: int
    n_dropped_missing: int
    comparison_role: str = "existing_referee_variant"
    left_object: str = ""
    right_object: str = ""
    left_transform: str = "identity"
    left_rank_type: str = "ordinary"
    right_rank_type: str = "ordinary"
    normalizer_alpha: float = 0.5
    pvalue_null_alpha: float = 2 / 3
    right_source_sha256: str = ""
    n_structural_zero: int = 0
    n_exact_net_zero_left: int = 0
    n_near_zero_left: int = 0
    n_good_left: Optional[int] = None
    n_bad_left: Optional[int] = None
    tie_method: str = "average"
    zero_sign_rule: str = "value >= 0 is non-improving"


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


def stable_seed(base_seed: int, horizon: int, model: str, comparison: str) -> int:
    payload = f"{base_seed}|{horizon}|{model}|{comparison}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def load_shapley_bundle(results_dir: Path, horizon: int):
    artifact = results_dir / f"shapleys_h{horizon}_upd.bin"
    if not artifact.exists():
        raise FileNotFoundError(f"Missing required artifact: {artifact}")

    with artifact.open("rb") as handle:
        bundle = pickle.load(handle)

    (pbsv_rmse, _pbsv_rmse_years), (_oshapley, _ishapley), oshapley_vi, ishapley_vi = bundle
    return pbsv_rmse, oshapley_vi, ishapley_vi


def shapley_bundle_sha256(results_dir: Path, horizon: int) -> str:
    artifact = results_dir / f"shapleys_h{horizon}_upd.bin"
    if not artifact.exists():
        raise FileNotFoundError(f"Missing required artifact: {artifact}")
    return sha256_file(artifact)


def load_insample_gpbsv(insample_gpbsv_dir: Path, horizon: int) -> pd.DataFrame:
    binary_path = insample_gpbsv_dir / f"insample_gpbsv_h{horizon}_v5.bin"
    csv_path = insample_gpbsv_dir / f"insample_gpbsv_h{horizon}_grouped.csv"
    if binary_path.exists():
        with binary_path.open("rb") as handle:
            payload = pickle.load(handle)
        grouped = payload["is_gpbsv_rmse_grouped"]
    elif csv_path.exists():
        grouped = pd.read_csv(csv_path, index_col=0)
    else:
        raise FileNotFoundError(
            f"Missing in-sample GPBSV for h={horizon}: expected {binary_path} or {csv_path}"
        )
    if not isinstance(grouped, pd.DataFrame):
        raise TypeError(f"In-sample GPBSV h={horizon} is not a DataFrame")
    return grouped


def load_structural_masks(mask_dir: Path, horizon: int) -> dict[str, pd.Series]:
    mask_path = mask_dir / f"structural_zero_mask_h{horizon}.csv"
    if not mask_path.exists():
        return {}

    mask = pd.read_csv(mask_path)
    required = {"model", "predictor", "structural_zero"}
    missing = required - set(mask.columns)
    if missing:
        raise ValueError(f"Structural-zero mask {mask_path} is missing columns: {sorted(missing)}")

    out: dict[str, pd.Series] = {}
    for model, group in mask.groupby("model", sort=False):
        out[str(model)] = pd.Series(
            group["structural_zero"].astype(bool).to_numpy(),
            index=group["predictor"].astype(str),
            name="structural_zero",
        )
    return out


def rank_importance(values: pd.Series) -> pd.Series:
    """Rank variable importance so larger importance receives a larger rank."""
    return values.rank(ascending=True, method="average")


def signed_rank_lower_loss_is_better(values: pd.Series) -> pd.Series:
    """
    Signed rank for RMSE GPBSV, matching Code/V002/iml_rev.py::rankdiffs2_w.

    Negative GPBSV values reduce RMSE and are beneficial, so they receive positive
    ranks. Non-negative values are harmful or non-improving and receive negative
    ranks.
    """
    ranks = pd.Series(0.0, index=values.index)

    good = values < 0
    bad = ~good

    if good.any():
        ranks.loc[good] = (-values.loc[good]).rank(ascending=True, method="average").to_numpy()
    if bad.any():
        ranks.loc[bad] = -values.loc[bad].rank(ascending=True, method="average").to_numpy()

    return ranks


def sum_of_squares(n: int) -> float:
    return n * (n + 1) * (2 * n + 1) / 6


def expected_signed_msdr_weighted(weights: np.ndarray, ranks: np.ndarray) -> float:
    """
    Expected weighted MSDR under the signed-rank null used by the paper code.

    This intentionally mirrors the current implementation's normalization, where
    the expected MSDR is based on a 50/50 beneficial/harmful signed-rank split.
    The p-value null can still use another alpha, as in the paper tables.
    """
    p = len(ranks)
    signed_rank_term = 0.0
    two_to_minus_p = 0.5**p

    for n_good in range(p + 1):
        prob = math.comb(p, n_good) * two_to_minus_p
        signed_rank_term += prob * (
            sum_of_squares(n_good) + sum_of_squares(p - n_good)
        )

    return float((np.sum(weights * (ranks**2)) + signed_rank_term) / p)


def expected_ordinary_msdr_weighted(weights: np.ndarray, ranks: np.ndarray) -> float:
    """
    Expected weighted MSDR for two independent unsigned rank permutations.
    """
    p = len(ranks)
    expected_rank = (p + 1) / 2
    expected_rank_sq = sum_of_squares(p) / p
    return float(
        np.mean(weights * (ranks**2 - 2 * ranks * expected_rank + expected_rank_sq))
    )


def monte_carlo_p_value(
    *,
    observed_mas: float,
    expected_msdr: float,
    ranks: np.ndarray,
    weights: np.ndarray,
    null_type: str,
    n_sims: int,
    alpha: float,
    rng: np.random.Generator,
    chunk_size: int,
    add_one: bool = False,
) -> float:
    """
    Estimate P(MAS_null >= MAS_observed) under unrelated ranks.

    The left ranks and weights are shuffled together, matching the current paper
    code's weighted signed-rank null. The right side is independently shuffled.
    """
    if n_sims <= 0:
        raise ValueError("n_sims must be positive")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    p = len(ranks)
    base_ranks = np.arange(1, p + 1, dtype=float)
    extreme_count = 0
    completed = 0

    while completed < n_sims:
        n_chunk = min(chunk_size, n_sims - completed)

        left_perm = np.argsort(rng.random((n_chunk, p)), axis=1)
        right_perm = np.argsort(rng.random((n_chunk, p)), axis=1)

        left_ranks = ranks[left_perm]
        left_weights = weights[left_perm]

        if null_type == "ordinary":
            right_ranks = base_ranks[right_perm]
        elif null_type == "signed":
            n_good = rng.binomial(p, alpha, size=n_chunk)
            col_idx = np.arange(p)[None, :]
            signed_template = np.where(
                col_idx < n_good[:, None],
                col_idx + 1,
                -(col_idx - n_good[:, None] + 1),
            ).astype(float)
            right_ranks = np.take_along_axis(signed_template, right_perm, axis=1)
        else:
            raise ValueError(f"Unknown null_type: {null_type}")

        msdr = np.mean(left_weights * (left_ranks - right_ranks) ** 2, axis=1)
        mas = 1 - msdr / expected_msdr
        extreme_count += int(np.sum(mas >= observed_mas))
        completed += n_chunk

    if add_one:
        return (extreme_count + 1) / (n_sims + 1)
    return extreme_count / n_sims


def rank_values(values: pd.Series, rank_type: str) -> pd.Series:
    if rank_type == "ordinary":
        return rank_importance(values)
    if rank_type == "signed":
        return signed_rank_lower_loss_is_better(values)
    raise ValueError(f"Unknown rank_type: {rank_type}")


def prepare_series(
    *,
    model: str,
    left: pd.DataFrame,
    right: pd.DataFrame,
    weights_source: pd.DataFrame,
    require_exact_columns: bool = False,
    drop_zero_weights: bool = True,
    structural_mask: Optional[pd.Series] = None,
) -> tuple[pd.Series, pd.Series, pd.Series, dict[str, int]]:
    """
    Align one model's left/right/weight series and drop non-ranked variables.
    """
    left_series = left.loc[model].copy()
    right_series = right.loc[model].copy()
    weights = weights_source.loc[model].copy()

    if require_exact_columns:
        if not left_series.index.equals(right_series.index):
            raise ValueError(f"Left/right predictor indexes differ for model={model}")
        if not left_series.index.equals(weights.index):
            raise ValueError(f"Left/weight predictor indexes differ for model={model}")
        all_columns = left_series.index
        n_dropped_missing = 0
    else:
        all_columns = left_series.index.intersection(right_series.index).intersection(weights.index)
        n_dropped_missing = (
            len(set(left_series.index) | set(right_series.index) | set(weights.index))
            - len(all_columns)
        )

    left_series = left_series.reindex(all_columns)
    right_series = right_series.reindex(all_columns)
    weights = weights.reindex(all_columns)

    n_dropped_base = 0
    if "base_contribution" in left_series.index:
        left_series = left_series.drop("base_contribution")
        right_series = right_series.drop("base_contribution")
        weights = weights.drop("base_contribution")
        n_dropped_base = 1

    valid = left_series.notna() & right_series.notna() & weights.notna()
    n_dropped_missing += int((~valid).sum())
    left_series = left_series.loc[valid]
    right_series = right_series.loc[valid]
    weights = weights.loc[valid]

    n_structural_zero = 0
    if structural_mask is not None:
        mask = structural_mask.reindex(left_series.index).fillna(False).astype(bool)
        n_structural_zero = int(mask.sum())
        left_series = left_series.loc[~mask]
        right_series = right_series.loc[~mask]
        weights = weights.loc[~mask]

    n_exact_net_zero_left = int((left_series.astype(float) == 0.0).sum())
    near_zero_left = np.isclose(left_series.astype(float), 0.0)
    n_near_zero_left = int((near_zero_left & (left_series.astype(float) != 0.0)).sum())

    n_dropped_zero_weight = 0
    if drop_zero_weights:
        positive_weight = ~np.isclose(weights.astype(float), 0.0)
        n_dropped_zero_weight = int((~positive_weight).sum())
        left_series = left_series.loc[positive_weight]
        right_series = right_series.loc[positive_weight]
        weights = weights.loc[positive_weight]

    if left_series.empty:
        raise ValueError(f"No ranked predictors remain for model={model}")

    return left_series, right_series, weights, {
        "n_dropped_base": n_dropped_base,
        "n_dropped_zero_weight": n_dropped_zero_weight,
        "n_dropped_missing": n_dropped_missing,
        "n_structural_zero": n_structural_zero,
        "n_exact_net_zero_left": n_exact_net_zero_left,
        "n_near_zero_left": n_near_zero_left,
    }


def compute_mas(
    *,
    horizon: int,
    model: str,
    comparison: str,
    comparison_role: str,
    left_object: str,
    right_object: str,
    left: pd.Series,
    right: pd.Series,
    weights: pd.Series,
    null_type: str,
    left_rank_type: str,
    right_rank_type: str,
    n_sims: int,
    alpha: float,
    seed: int,
    rng: np.random.Generator,
    chunk_size: int,
    dropped: dict[str, int],
    right_source_sha256: str = "",
    left_transform: str = "identity",
    add_one_pvalue: bool = False,
) -> MASResult:
    ranks_left = rank_values(left, left_rank_type).to_numpy(dtype=float)
    ranks_right = rank_values(right, right_rank_type).to_numpy(dtype=float)
    weights_array = (weights / weights.mean()).to_numpy(dtype=float)
    if not np.isfinite(weights_array).all():
        raise ValueError(f"Invalid MAS weights for h={horizon}, model={model}, comparison={comparison}")

    if null_type == "ordinary":
        expected_msdr = expected_ordinary_msdr_weighted(weights_array, ranks_left)
        n_good = None
        n_bad = None
    elif null_type == "signed":
        expected_msdr = expected_signed_msdr_weighted(weights_array, ranks_left)
        n_good = int((right < 0).sum())
        n_bad = int((right >= 0).sum())
    else:
        raise ValueError(f"Unknown null_type: {null_type}")

    msdr = float(np.mean(weights_array * (ranks_left - ranks_right) ** 2))
    mas = float(1 - msdr / expected_msdr)
    p_value = monte_carlo_p_value(
        observed_mas=mas,
        expected_msdr=expected_msdr,
        ranks=ranks_left,
        weights=weights_array,
        null_type=null_type,
        n_sims=n_sims,
        alpha=alpha,
        rng=rng,
        chunk_size=chunk_size,
        add_one=add_one_pvalue,
    )

    return MASResult(
        horizon=horizon,
        model=model,
        model_label=MODEL_LABELS[model],
        comparison=comparison,
        comparison_label=COMPARISON_LABELS[comparison],
        weighting_source=weights.name or "unknown",
        null_type=null_type,
        mas=mas,
        p_value=float(p_value),
        msdr=msdr,
        expected_msdr=expected_msdr,
        null_alpha=alpha,
        n_sims=n_sims,
        seed=seed,
        n_predictors=len(left),
        n_good_gpbsv=n_good,
        n_bad_gpbsv=n_bad,
        n_dropped_base=dropped["n_dropped_base"],
        n_dropped_zero_weight=dropped["n_dropped_zero_weight"],
        n_dropped_missing=dropped["n_dropped_missing"],
        comparison_role=comparison_role,
        left_object=left_object,
        right_object=right_object,
        left_transform=left_transform,
        left_rank_type=left_rank_type,
        right_rank_type=right_rank_type,
        normalizer_alpha=0.5 if null_type == "signed" else np.nan,
        pvalue_null_alpha=alpha,
        right_source_sha256=right_source_sha256,
        n_structural_zero=dropped.get("n_structural_zero", 0),
        n_exact_net_zero_left=dropped.get("n_exact_net_zero_left", 0),
        n_near_zero_left=dropped.get("n_near_zero_left", 0),
        n_good_left=int((left < 0).sum()) if left_rank_type == "signed" else None,
        n_bad_left=int((left >= 0).sum()) if left_rank_type == "signed" else None,
    )


def comparison_specs(
    include_sensitivity: bool,
    include_insample_gpbsv: bool,
) -> list[dict[str, object]]:
    specs = [
        {
            "comparison": "is_shapley_vs_oos_gpbsv",
            "left": "ishapley_vi",
            "right": "pbsv_rmse",
            "weights": "ishapley_vi",
            "null_type": "signed",
            "comparison_role": "existing_referee_variant",
            "left_rank_type": "ordinary",
            "right_rank_type": "signed",
            "left_transform": "identity",
            "drop_zero_weights": True,
            "require_exact_columns": False,
        },
        {
            "comparison": "is_shapley_vs_oos_shapley",
            "left": "ishapley_vi",
            "right": "oshapley_vi",
            "weights": "ishapley_vi",
            "null_type": "ordinary",
            "comparison_role": "existing_referee_variant",
            "left_rank_type": "ordinary",
            "right_rank_type": "ordinary",
            "left_transform": "identity",
            "drop_zero_weights": True,
            "require_exact_columns": False,
        },
        {
            "comparison": "oos_shapley_vs_oos_gpbsv",
            "left": "oshapley_vi",
            "right": "pbsv_rmse",
            "weights": "oshapley_vi",
            "null_type": "signed",
            "comparison_role": "existing_referee_variant",
            "left_rank_type": "ordinary",
            "right_rank_type": "signed",
            "left_transform": "identity",
            "drop_zero_weights": True,
            "require_exact_columns": False,
        },
    ]
    if include_sensitivity:
        specs.append({
            "comparison": "oos_shapley_vs_oos_gpbsv_is_weighted",
            "left": "oshapley_vi",
            "right": "pbsv_rmse",
            "weights": "ishapley_vi",
            "null_type": "signed",
            "comparison_role": "existing_referee_variant",
            "left_rank_type": "ordinary",
            "right_rank_type": "signed",
            "left_transform": "identity",
            "drop_zero_weights": True,
            "require_exact_columns": False,
        })
    if include_insample_gpbsv:
        specs.extend([
            {
                "comparison": "is_gpbsv_abs_vs_saved_oos_gpbsv_signed",
                "left": "is_gpbsv_abs",
                "right": "pbsv_rmse",
                "weights": "is_gpbsv_abs",
                "null_type": "signed",
                "comparison_role": "primary_mas_style_robustness",
                "left_rank_type": "ordinary",
                "right_rank_type": "signed",
                "left_transform": "abs",
                "drop_zero_weights": False,
                "require_exact_columns": True,
            },
            {
                "comparison": "is_gpbsv_signed_vs_saved_oos_gpbsv_signed",
                "left": "is_gpbsv_signed",
                "right": "pbsv_rmse",
                "weights": "is_gpbsv_abs",
                "null_type": "signed",
                "comparison_role": "signed_concordance_companion",
                "left_rank_type": "signed",
                "right_rank_type": "signed",
                "left_transform": "identity",
                "drop_zero_weights": False,
                "require_exact_columns": True,
            },
        ])
    return specs


def compute_referee_mas_variants(
    *,
    results_dir: Path,
    insample_gpbsv_dir: Path,
    structural_mask_dir: Path,
    horizons: Iterable[int],
    models: Iterable[str],
    n_sims: int,
    alpha: float,
    seed: int,
    chunk_size: int,
    include_sensitivity: bool,
    include_insample_gpbsv: bool,
    equal_weights: bool = False,
) -> pd.DataFrame:
    rows: list[MASResult] = []
    legacy_rng = np.random.default_rng(seed)

    for horizon in horizons:
        pbsv_rmse, oshapley_vi, ishapley_vi = load_shapley_bundle(results_dir, horizon)
        right_source_sha256 = shapley_bundle_sha256(results_dir, horizon)
        structural_masks: dict[str, pd.Series] = {}
        objects = {
            "pbsv_rmse": pbsv_rmse,
            "oshapley_vi": oshapley_vi,
            "ishapley_vi": ishapley_vi,
        }
        if include_insample_gpbsv:
            is_gpbsv_signed = load_insample_gpbsv(insample_gpbsv_dir, horizon)
            if not is_gpbsv_signed.columns.equals(pbsv_rmse.columns):
                raise ValueError(f"IS GPBSV columns do not match saved OOS GPBSV columns for h={horizon}")
            objects["is_gpbsv_signed"] = is_gpbsv_signed
            objects["is_gpbsv_abs"] = is_gpbsv_signed.abs()
            structural_masks = load_structural_masks(structural_mask_dir, horizon)

        for spec in comparison_specs(include_sensitivity, include_insample_gpbsv):
            is_new_insample_comparison = str(spec["comparison"]).startswith("is_gpbsv_")
            for model in models:
                if model not in objects[spec["left"]].index:
                    if include_insample_gpbsv and str(spec["left"]).startswith("is_gpbsv"):
                        continue
                    raise KeyError(f"Model {model} missing from {spec['left']} for h={horizon}")
                left, right, weights, dropped = prepare_series(
                    model=model,
                    left=objects[spec["left"]],
                    right=objects[spec["right"]],
                    weights_source=objects[spec["weights"]],
                    require_exact_columns=bool(spec.get("require_exact_columns", False)),
                    drop_zero_weights=bool(spec.get("drop_zero_weights", True)),
                    structural_mask=(
                        structural_masks.get(model) if is_new_insample_comparison else None
                    ),
                )
                weights.name = spec["weights"]
                if equal_weights:
                    # Equal weights AFTER the structural-zero drop: never-selected
                    # predictors stay excluded, surviving predictors weigh equally.
                    weights = pd.Series(1.0, index=weights.index,
                                        name=f"equal(drop via {spec['weights']})")
                if is_new_insample_comparison:
                    row_seed = stable_seed(seed, int(horizon), model, str(spec["comparison"]))
                    row_rng = np.random.default_rng(row_seed)
                    add_one_pvalue = True
                else:
                    row_seed = seed
                    row_rng = legacy_rng
                    add_one_pvalue = False
                rows.append(compute_mas(
                    horizon=horizon,
                    model=model,
                    comparison=spec["comparison"],
                    comparison_role=str(spec["comparison_role"]),
                    left_object=str(spec["left"]),
                    right_object=str(spec["right"]),
                    left=left,
                    right=right,
                    weights=weights,
                    null_type=spec["null_type"],
                    left_rank_type=str(spec["left_rank_type"]),
                    right_rank_type=str(spec["right_rank_type"]),
                    n_sims=n_sims,
                    alpha=alpha,
                    seed=row_seed,
                    rng=row_rng,
                    chunk_size=chunk_size,
                    dropped=dropped,
                    right_source_sha256=right_source_sha256 if spec["right"] == "pbsv_rmse" else "",
                    left_transform=str(spec["left_transform"]),
                    add_one_pvalue=add_one_pvalue,
                ))

    return pd.DataFrame([row.__dict__ for row in rows])


def pivot_metric(details: pd.DataFrame, metric: str) -> pd.DataFrame:
    table = details.pivot_table(
        index=["comparison_label", "model_label"],
        columns="horizon",
        values=metric,
        aggfunc="first",
    )
    return table.reindex(
        pd.MultiIndex.from_product(
            [
                [COMPARISON_LABELS[key] for key in COMPARISON_LABELS if key in set(details["comparison"])],
                [MODEL_LABELS[key] for key in MODEL_KEYS],
            ],
            names=["comparison_label", "model_label"],
        )
    )


def pvalue_stars(p_value: float) -> str:
    if p_value <= 0.01:
        return "***"
    if p_value <= 0.05:
        return "**"
    if p_value <= 0.10:
        return "*"
    return ""


def format_mas_with_stars(details: pd.DataFrame) -> pd.DataFrame:
    formatted = details.copy()
    formatted["value"] = [
        f"{mas:.3f}{pvalue_stars(pval)}"
        for mas, pval in zip(formatted["mas"], formatted["p_value"])
    ]
    return pivot_metric(formatted, "value")


def write_outputs(details: pd.DataFrame, output_dir: Path, alpha: float, n_sims: int,
                  equal_weights: bool = False) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"a{alpha:.2f}_mc{n_sims}"
    if equal_weights:
        suffix += "_ew"

    has_insample_gpbsv = details["comparison"].astype(str).str.startswith("is_gpbsv_").any()
    prefix = "mas_is_oos_gpbsv_v5" if has_insample_gpbsv else "mas_referee_variants"

    details_csv = output_dir / f"{prefix}_details_{suffix}.csv"
    mas_csv = output_dir / f"{prefix}_{suffix}.csv"
    pvalues_csv = output_dir / f"{prefix}_pvalues_{suffix}.csv"
    formatted_csv = output_dir / f"{prefix}_formatted_{suffix}.csv"
    excel_path = output_dir / f"{prefix}_{suffix}.xlsx"
    latex_path = output_dir / f"{prefix}_{suffix}.tex"

    mas_table = pivot_metric(details, "mas")
    pvalue_table = pivot_metric(details, "p_value")
    formatted_table = format_mas_with_stars(details)
    n_predictors_table = pivot_metric(details, "n_predictors")
    n_good_table = pivot_metric(details, "n_good_gpbsv")

    details.to_csv(details_csv, index=False)
    mas_table.to_csv(mas_csv)
    pvalue_table.to_csv(pvalues_csv)
    formatted_table.to_csv(formatted_csv)

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        formatted_table.to_excel(writer, sheet_name="mas_formatted")
        mas_table.to_excel(writer, sheet_name="mas")
        pvalue_table.to_excel(writer, sheet_name="pvalues")
        n_predictors_table.to_excel(writer, sheet_name="n_predictors")
        n_good_table.to_excel(writer, sheet_name="n_good_gpbsv")
        details.to_excel(writer, sheet_name="details", index=False)

    with latex_path.open("w", encoding="utf-8") as handle:
        handle.write(formatted_table.to_latex(escape=True, na_rep=""))

    return {
        "details_csv": details_csv,
        "mas_csv": mas_csv,
        "pvalues_csv": pvalues_csv,
        "formatted_csv": formatted_csv,
        "excel": excel_path,
        "latex": latex_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute standalone MAS variants for the referee-response appendix."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help=f"Directory containing shapleys_h*_upd.bin artifacts. Default: {repo_relative(DEFAULT_RESULTS_DIR)}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory. Default: {repo_relative(DEFAULT_OUTPUT_DIR)}",
    )
    parser.add_argument(
        "--insample-gpbsv-dir",
        type=Path,
        default=DEFAULT_INSAMPLE_GPBSV_DIR,
        help=(
            "Directory containing insample_gpbsv_h{h}_grouped.csv or "
            f"insample_gpbsv_h{{h}}_v5.bin. Default: {repo_relative(DEFAULT_INSAMPLE_GPBSV_DIR)}"
        ),
    )
    parser.add_argument(
        "--structural-mask-dir",
        type=Path,
        default=DEFAULT_INSAMPLE_GPBSV_DIR,
        help=(
            "Directory containing structural_zero_mask_h{h}.csv. "
            f"Default: {repo_relative(DEFAULT_INSAMPLE_GPBSV_DIR)}"
        ),
    )
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=list(HORIZONS),
        help="Forecast horizons to process.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MODEL_KEYS),
        choices=list(MODEL_KEYS),
        help="Model keys to process.",
    )
    parser.add_argument(
        "--n-sims",
        type=int,
        default=1_000_000,
        help="Monte Carlo draws per horizon/model/comparison.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=2 / 3,
        help="Signed-rank null probability that a GPBSV contribution is beneficial.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260602,
        help="Random seed for reproducible Monte Carlo p-values.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=20_000,
        help="Monte Carlo chunk size. Reduce if memory is constrained.",
    )
    parser.add_argument(
        "--include-sensitivity",
        action="store_true",
        help="Also compute OOS Shapley VI vs OOS GPBSV with IS Shapley weights.",
    )
    parser.add_argument(
        "--include-insample-gpbsv",
        action="store_true",
        help="Also compute the v5 in-sample GPBSV MAS-style and signed-concordance comparisons.",
    )
    parser.add_argument(
        "--equal-weights",
        action="store_true",
        help="Use equal weights (after the structural-zero drop) instead of "
             "VI-proportional weights. Output files get an _ew suffix.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 <= args.alpha <= 1:
        raise ValueError("--alpha must be between 0 and 1")

    details = compute_referee_mas_variants(
        results_dir=args.results_dir,
        insample_gpbsv_dir=args.insample_gpbsv_dir,
        structural_mask_dir=args.structural_mask_dir,
        horizons=args.horizons,
        models=args.models,
        n_sims=args.n_sims,
        alpha=args.alpha,
        seed=args.seed,
        chunk_size=args.chunk_size,
        include_sensitivity=args.include_sensitivity,
        include_insample_gpbsv=args.include_insample_gpbsv,
        equal_weights=args.equal_weights,
    )
    outputs = write_outputs(details, args.output_dir, args.alpha, args.n_sims,
                            equal_weights=args.equal_weights)

    print("Computed MAS referee variants")
    print(f"Rows: {len(details)}")
    print(f"Monte Carlo draws per row: {args.n_sims}")
    for name, path in outputs.items():
        print(f"{name}: {repo_relative(path)}")


if __name__ == "__main__":
    main()
